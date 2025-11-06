# Project Plan

## 4.4 Orchestrator (app/orchestrator.py)
### Step 2: Parallel Classification
- Call MySQLRepository.get_template_metadata(task_type)
  - Returns: {template_id, task_type, version, minio_path, variables, description}
  
- **Call MinIOClient.get_template_content(minio_path)** ← **Uses Infrastructure MinIO client**
  - Returns: Template content string

## 4.5 Service Layer Components
### Task Services

**Task Decisioner (app/services/task/decisioner.py)**
- **RECEIVES flow_type from Flow Decisioner as input**
- For fixed flows: Return task immediately
- For general_chat: Perform similarity search + LLM re-ranking
- Returns: {task_type, confidence, reason}
- **Receives Weaviate client and LLM client via dependency injection**

**MySQL Repository (app/services/task/mysql_repo.py)**
- Manage prompt template metadata in MySQL
- CRUD operations for templates
- Query templates by task_type
- Returns template metadata including minio_path
- **Receives MySQL client via dependency injection**

**Template Loading:**
- Task services DO NOT have a separate MinIO client
- Use Infrastructure MinIO client (app/infrastructure/minio_client.py)
- Orchestrator calls Infrastructure MinIO client directly to load templates
- Template loading happens in Orchestrator, not in Task services

## 5.1 Request Flow
### Step 5
5. **Step 2: Parallel Processing**
   - Orchestrator calls FlowDecisioner.classify_flow()
     - Applies rule-based detection
     - Falls back to LLM if needed
     - Returns: {flow_type: str, confidence: float, reason: str}
   - Orchestrator calls TaskDecisioner.detect_task(user_input, flow_type)
     - **Receives flow_type as parameter**
     - For fixed flows: Returns immediately
     - For general_chat: Performs Weaviate search + LLM re-ranking
     - Returns: {task_type: str, confidence: float, reason: str}
   - Orchestrator calls MySQLRepository.get_template_metadata(task_type)
     - Queries MySQL for template info
     - Returns: {template_id, task_type, version, minio_path, variables, description}
   - **Orchestrator calls Infrastructure MinIOClient.get_template_content(minio_path)**
     - Downloads template from MinIO using shared Infrastructure client
     - Returns: Template content string
   - Orchestrator calls RAGRetriever.should_use_rag()
     - Decides if RAG needed
     - If yes: Retrieves chunks from Weaviate
   - Orchestrator calls ToolExecutor.should_execute_tool()
     - Decides if tool execution needed
     - If yes: Executes tool and returns result

