# Task Decisioner Component - Detailed Implementation Plan

## Document Purpose
This plan provides comprehensive specifications for implementing the Task Decisioner component of the LLM Orchestrator system. Use this document with VS Code Copilot to generate code that follows the exact architecture defined in PROJECT_PLAN_Version3.md and the task_decisioner.xml diagram.

---

## 1. Component Overview

### 1.1 Purpose
The Task Decisioner is responsible for determining the specific task type that best matches the user's input. It receives the `flow_type` from the Flow Decisioner as a critical input parameter and uses different detection strategies based on the flow type.

### 1.2 Key Responsibilities
- Receive `flow_type` from Flow Decisioner as input parameter
- For fixed flows (file_analysis, image_generation): Return task type immediately without search
- For general_chat flow: Perform dynamic task detection using Weaviate similarity search
- Use LLM re-ranking to select the best matching task from candidates
- Return task type with confidence score and reasoning
- Handle low-confidence scenarios with fallback to general_chat

### 1.3 Integration Points
- **Input**: Receives `flow_type` from Flow Decisioner (string parameter)
- **Dependencies**: 
  - Weaviate client (for task similarity search)
  - LLM client (for embedding generation and re-ranking)
- **Output**: Task type with confidence score and reason

---

## 2. File Location and Structure

### 2.1 Main Component File
**File Path**: `app/services/task/decisioner.py`

**Class Name**: `TaskDecisioner`

**Dependencies to Import**:
- WeaviateClient from `app/infrastructure/weaviate_client.py`
- LLMClient from `app/services/chat/llm_client.py`
- Logger from `app/utils/logger.py`
- Constants from `app/utils/constants.py` (FlowType enum, TaskType enum, confidence thresholds)
- Custom exceptions from `app/utils/exceptions.py`
- tiktoken for token counting

### 2.2 Supporting Files

**File Path**: `app/utils/constants.py`
- Add FlowType enum: GENERAL_CHAT, FILE_ANALYSIS, IMAGE_GENERATION
- Add TaskType enum: List of all supported task types
- Add TASK_SIMILARITY_THRESHOLD: Default 0.7
- Add TASK_TOP_K: Default 5 (number of candidates to retrieve)
- Add MIN_SIMILARITY_SCORE: Default 0.3

**File Path**: `app/schemas/task.py` (new file)
- Create Pydantic models for task detection request/response
- TaskDetectionRequest: user_input, flow_type
- TaskDetectionResponse: task_type, confidence, reason, candidates (optional)

---

## 3. Class Structure

### 3.1 TaskDecisioner Class

**Constructor Parameters**:
- `weaviate_client`: WeaviateClient instance (injected)
- `llm_client`: LLMClient instance (injected)
- `logger`: Logger instance (injected)
- `config`: Configuration dictionary with thresholds

**Instance Variables**:
- `self.weaviate_client`: Stored Weaviate client
- `self.llm_client`: Stored LLM client
- `self.logger`: Stored logger
- `self.similarity_threshold`: Float (from config or default 0.7)
- `self.top_k`: Integer (from config or default 5)
- `self.min_similarity`: Float (from config or default 0.3)
- `self.weaviate_task_class`: String (class name in Weaviate, default "TaskType")

**Public Methods**:

1. `detect_task(user_input: str, flow_type: str) -> dict`
   - Main entry point for task detection
   - Receives flow_type as critical parameter
   - Routes to appropriate detection method based on flow_type
   - Returns: {task_type: str, confidence: float, reason: str}

2. `detect_task_for_general_chat(user_input: str) -> dict`
   - Performs dynamic task detection for general_chat flow
   - Generates embedding for user input
   - Searches Weaviate for similar tasks
   - Performs LLM re-ranking
   - Checks confidence threshold
   - Returns task type with confidence and reason

**Private/Helper Methods**:

3. `_is_fixed_flow(flow_type: str) -> bool`
   - Checks if flow_type is file_analysis or image_generation
   - Returns boolean

4. `_get_fixed_task_type(flow_type: str) -> dict`
   - Returns fixed task type for non-general-chat flows
   - For FILE_ANALYSIS: Returns {task_type: "file_analysis", confidence: 1.0, reason: "Fixed task for file analysis flow"}
   - For IMAGE_GENERATION: Returns {task_type: "image_generation", confidence: 1.0, reason: "Fixed task for image generation flow"}

5. `_generate_embedding(text: str) -> list[float]`
   - Calls LLMClient.embed(text)
   - Returns embedding vector
   - Handles errors and retries
   - Logs embedding generation

6. `_search_similar_tasks(embedding_vector: list[float]) -> list[dict]`
   - Performs Weaviate near_vector search
   - Query parameters:
     - vector: embedding_vector
     - limit: self.top_k
     - return_properties: ["name", "description"]
     - return_metadata: ["certainty", "distance"]
   - Returns list of candidates: [{name: str, description: str, certainty: float}]
   - Filters results with certainty >= self.min_similarity
   - Sorts by certainty descending

7. `_rerank_with_llm(user_input: str, candidates: list[dict]) -> dict`
   - Formats candidates into prompt
   - Calls LLM to select best matching task
   - Parses LLM response (JSON mode)
   - Returns: {task_type: str, confidence: float, reason: str}
   - Handles parsing errors with fallback

8. `_check_confidence_threshold(confidence: float) -> bool`
   - Compares confidence with self.similarity_threshold
   - Returns True if confidence >= threshold

9. `_fallback_to_general_chat(reason: Optional[str] = None) -> dict`
   - Returns default general_chat task
   - Returns: {task_type: "general_chat", confidence: 1.0, reason: "Fallback due to low confidence in task detection"}

10. `_format_candidates_for_llm(candidates: list[dict]) -> str`
    - Formats candidate list into readable prompt text
    - Returns formatted string for LLM consumption

11. `_log_detection_result(user_input: str, flow_type: str, result: dict) -> None`
    - Logs task detection with structured format
    - Includes: user_input (truncated), flow_type, task_type, confidence, reason
    - Log level: INFO for successful detection, WARNING for fallback

---

## 4. Detection Flow Logic

### 4.1 Main Detection Flow (detect_task method)

**Step 1: Validate Inputs**
- Check if user_input is non-empty string
- Check if flow_type is valid enum value
- Raise ValidationError if invalid
- Log incoming request with correlation ID

**Step 2: Check Flow Type**
- Call `_is_fixed_flow(flow_type)`
- If True: Branch to fixed task detection
- If False: Branch to dynamic task detection

**Step 3a: Fixed Task Detection (for file_analysis and image_generation)**
- Call `_get_fixed_task_type(flow_type)`
- Log result with flow_type
- Return immediately with confidence = 1.0
- Processing time: < 10ms

**Step 3b: Dynamic Task Detection (for general_chat)**
- Call `detect_task_for_general_chat(user_input)`
- Follow dynamic detection flow (see 4.2)
- Processing time: ~1.5-2 seconds

**Step 4: Return Result**
- Format response dictionary
- Log detection result
- Return to Orchestrator

### 4.2 Dynamic Task Detection Flow (detect_task_for_general_chat method)

**Step 1: Generate Embedding**
- Call `_generate_embedding(user_input)`
- Use same embedding model as task registration
- Handle embedding API errors
- Log embedding generation time

**Step 2: Search Weaviate**
- Call `_search_similar_tasks(embedding_vector)`
- Retrieve top K candidates (default 5)
- Filter by minimum similarity threshold
- Log number of candidates found

**Step 3: Handle No Candidates**
- If candidates list is empty or all below min_similarity:
  - Log warning: "No suitable tasks found, falling back to general_chat"
  - Call `_fallback_to_general_chat()`
  - Return fallback result

**Step 4: LLM Re-ranking**
- If candidates found:
  - Call `_rerank_with_llm(user_input, candidates)`
  - LLM examines all candidates
  - Selects best match with confidence score
  - Provides reasoning

**Step 5: Confidence Check**
- Call `_check_confidence_threshold(confidence)`
- If confidence >= threshold (default 0.7):
  - Accept LLM's selection
  - Log successful detection
  - Return result with task_type, confidence, reason
- If confidence < threshold:
  - Log low confidence warning
  - Call `_fallback_to_general_chat()`
  - Return fallback result

### 4.3 Fallback Strategy

**When to Trigger Fallback**:
- No candidates found in Weaviate search
- All candidates below minimum similarity threshold
- LLM re-ranking confidence below threshold
- LLM API errors or timeout
- Unexpected exceptions during detection

**Fallback Behavior**:
- Return task_type = "general_chat"
- Set confidence = 1.0 (confident in fallback)
- Provide clear reason explaining why fallback was used
- Log warning with details
- Continue processing (graceful degradation)

---

## 5. Weaviate Integration

### 5.1 Schema Requirements

**Weaviate Class Name**: `TaskType`

**Properties**:
- `name`: text (task type identifier, e.g., "translate", "summarize_article")
- `description`: text (detailed description of what the task does)

**Vectorizer**: none (we provide embedding vectors manually)

**Index Configuration**:
- Vector index type: HNSW (Hierarchical Navigable Small World)
- Distance metric: cosine similarity

### 5.2 Search Query Structure

**Method**: `near_vector` search

**Query Parameters**:
```python
{
    "vector": embedding_vector,  # Query embedding from user input
    "limit": self.top_k,  # Default 5
    "certainty": self.min_similarity,  # Minimum 0.3
    "return_properties": ["name", "description"],
    "return_metadata": ["certainty", "distance"]
}
```

**Response Structure**:
```python
[
    {
        "name": "translate",
        "description": "Translate text from one language to another",
        "certainty": 0.92,  # Similarity score (0-1)
        "distance": 0.08    # Distance metric
    },
    {
        "name": "explain_concept",
        "description": "Explain complex concepts in simple terms",
        "certainty": 0.65,
        "distance": 0.35
    }
    # ... more candidates
]
```

### 5.3 Error Handling for Weaviate

**Connection Errors**:
- Catch Weaviate connection exceptions
- Log error with context
- Retry once with exponential backoff
- If still fails: Call fallback_to_general_chat()
- Raise MemoryError with descriptive message

**Query Errors**:
- Catch invalid query exceptions
- Log error details
- Fall back to general_chat
- Don't crash the pipeline

**Empty Results**:
- Not an error condition
- Log warning: "No tasks found matching criteria"
- Proceed to fallback

---

## 6. LLM Integration

### 6.1 Embedding Generation

**Method**: Call `LLMClient.embed(text)`

**Model**: Use same embedding model as task registration (consistency required)
- OpenAI: "text-embedding-3-small" or "text-embedding-ada-002"
- Hugging Face alternative: "sentence-transformers/all-MiniLM-L6-v2"

**Input Processing**:
- Clean user input (remove extra whitespace)
- Truncate if necessary (max tokens for embedding model)
- Handle special characters

**Error Handling**:
- Catch API timeout errors
- Retry once after 2 seconds
- If fails: Log error and raise LLMError
- Don't proceed without embedding

### 6.2 LLM Re-ranking

**Model**: GPT-4o-mini (fast and cost-effective)

**Prompt Template**:
```
You are an expert task classifier for a chatbot system. Your job is to select the most appropriate task type based on the user's input and a list of candidate tasks.

**User Input:**
{user_input}

**Candidate Tasks:**
{formatted_candidates}

**Instructions:**
1. Analyze the user's input carefully
2. Compare it with each candidate task description
3. Consider the similarity scores but use your judgment
4. Select the single most appropriate task
5. Provide a confidence score (0.0-1.0) for your selection
6. Explain your reasoning briefly

**Response Format (JSON):**
{{
  "selected_task": "task_name",
  "confidence": 0.85,
  "reason": "The user's input matches this task because..."
}}

Respond ONLY with valid JSON.
```

**Candidate Formatting**:
```
1. translate (similarity: 0.92)
   Description: Translate text from one language to another
   
2. explain_concept (similarity: 0.65)
   Description: Explain complex concepts in simple terms
   
3. general_chat (similarity: 0.55)
   Description: Casual conversation and small talk
```

**LLM Configuration**:
- Temperature: 0.3 (low for consistency)
- Max tokens: 200 (enough for JSON response)
- Response format: JSON mode (structured output)
- Top_p: 0.9
- Frequency penalty: 0

**Response Parsing**:
- Parse JSON response
- Validate required fields: selected_task, confidence, reason
- Validate confidence is float between 0.0 and 1.0
- Validate selected_task exists in candidates
- Handle parsing errors gracefully

**Error Handling**:
- If LLM returns invalid JSON: Log error, use highest similarity candidate
- If selected_task not in candidates: Use highest similarity candidate
- If confidence out of range: Clamp to 0.0-1.0
- If API timeout: Retry once, then fallback
- If API error: Log and fallback to general_chat

---

## 7. Configuration

### 7.1 Environment Variables (add to .env)

```bash
# Task Decisioner Configuration
TASK_SIMILARITY_THRESHOLD=0.7
TASK_TOP_K=5
TASK_MIN_SIMILARITY=0.3
WEAVIATE_TASK_CLASS=TaskType

# Embedding Model
TASK_EMBEDDING_MODEL=text-embedding-3-small

# Re-ranking Model
TASK_RERANKING_MODEL=gpt-4o-mini
TASK_RERANKING_TEMPERATURE=0.3
TASK_RERANKING_MAX_TOKENS=200
```

### 7.2 Configuration Class (app/config.py additions)

Add these properties to the Config class:
```python
# Task Decisioner Settings
TASK_SIMILARITY_THRESHOLD = float(os.getenv('TASK_SIMILARITY_THRESHOLD', 0.7))
TASK_TOP_K = int(os.getenv('TASK_TOP_K', 5))
TASK_MIN_SIMILARITY = float(os.getenv('TASK_MIN_SIMILARITY', 0.3))
WEAVIATE_TASK_CLASS = os.getenv('WEAVIATE_TASK_CLASS', 'TaskType')

# Embedding Configuration
TASK_EMBEDDING_MODEL = os.getenv('TASK_EMBEDDING_MODEL', 'text-embedding-3-small')

# Re-ranking Configuration
TASK_RERANKING_MODEL = os.getenv('TASK_RERANKING_MODEL', 'gpt-4o-mini')
TASK_RERANKING_TEMPERATURE = float(os.getenv('TASK_RERANKING_TEMPERATURE', 0.3))
TASK_RERANKING_MAX_TOKENS = int(os.getenv('TASK_RERANKING_MAX_TOKENS', 200))
```

### 7.3 Validation

Add validation in Config.validate():
```python
# Validate threshold ranges
if not 0.0 <= Config.TASK_SIMILARITY_THRESHOLD <= 1.0:
    raise ValueError("TASK_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")

if not 0.0 <= Config.TASK_MIN_SIMILARITY <= 1.0:
    raise ValueError("TASK_MIN_SIMILARITY must be between 0.0 and 1.0")

if Config.TASK_TOP_K < 1:
    raise ValueError("TASK_TOP_K must be at least 1")
```

---

## 8. Orchestrator Integration

### 8.1 Initialization in Orchestrator

**In `app/orchestrator.py` constructor**:

```python
# Initialize TaskDecisioner
self.task_decisioner = TaskDecisioner(
    weaviate_client=weaviate_client,
    llm_client=llm_client,
    logger=self.logger,
    config={
        'similarity_threshold': Config.TASK_SIMILARITY_THRESHOLD,
        'top_k': Config.TASK_TOP_K,
        'min_similarity': Config.TASK_MIN_SIMILARITY,
        'weaviate_task_class': Config.WEAVIATE_TASK_CLASS
    }
)
```

### 8.2 Usage in Pipeline (process_message method)

**Step 2: Parallel Classification**

```python
# Step 2a: Classify flow type (must happen first)
flow_result = self.flow_decisioner.classify_flow(
    sanitized_input, 
    last_intent, 
    attachments
)
flow_type = flow_result['flow_type']
flow_confidence = flow_result['confidence']

# Step 2b: Detect task type (receives flow_type as parameter)
task_result = self.task_decisioner.detect_task(
    user_input=sanitized_input,
    flow_type=flow_type  # Critical parameter passed here
)
task_type = task_result['task_type']
task_confidence = task_result['confidence']
task_reason = task_result['reason']

# Log the flow → task relationship
self.logger.info(
    "Flow and Task Detection Complete",
    extra={
        'flow_type': flow_type,
        'flow_confidence': flow_confidence,
        'task_type': task_type,
        'task_confidence': task_confidence,
        'task_reason': task_reason
    }
)
```

### 8.3 Error Handling in Orchestrator

```python
try:
    task_result = self.task_decisioner.detect_task(sanitized_input, flow_type)
except ValidationError as e:
    self.logger.error(f"Task detection validation error: {e}")
    # Use fallback
    task_result = {
        'task_type': 'general_chat',
        'confidence': 1.0,
        'reason': 'Fallback due to validation error'
    }
except LLMError as e:
    self.logger.error(f"LLM error during task detection: {e}")
    # Use fallback
    task_result = {
        'task_type': 'general_chat',
        'confidence': 1.0,
        'reason': 'Fallback due to LLM error'
    }
except MemoryError as e:
    self.logger.error(f"Weaviate error during task detection: {e}")
    # Use fallback
    task_result = {
        'task_type': 'general_chat',
        'confidence': 1.0,
        'reason': 'Fallback due to database error'
    }
except Exception as e:
    self.logger.error(f"Unexpected error during task detection: {e}")
    raise OrchestrationError(f"Task detection failed: {e}")
```

---

## 9. Response Schema

### 9.1 TaskDetectionResponse Schema

**File**: `app/schemas/task.py`

**Pydantic Model**:
```python
from pydantic import BaseModel, Field
from typing import Optional, List

class TaskCandidate(BaseModel):
    name: str = Field(..., description="Task type name")
    description: str = Field(..., description="Task description")
    certainty: float = Field(..., ge=0.0, le=1.0, description="Similarity score")

class TaskDetectionResponse(BaseModel):
    task_type: str = Field(..., description="Selected task type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in selection")
    reason: str = Field(..., description="Explanation for selection")
    flow_type: str = Field(..., description="Input flow type that influenced detection")
    candidates: Optional[List[TaskCandidate]] = Field(None, description="Top candidate tasks (only for general_chat)")
    
    class Config:
        schema_extra = {
            "example": {
                "task_type": "translate",
                "confidence": 0.92,
                "reason": "User explicitly requested translation from English to Spanish",
                "flow_type": "general_chat",
                "candidates": [
                    {
                        "name": "translate",
                        "description": "Translate text from one language to another",
                        "certainty": 0.92
                    },
                    {
                        "name": "explain_concept",
                        "description": "Explain complex concepts",
                        "certainty": 0.65
                    }
                ]
            }
        }
```

### 9.2 Return Value from detect_task Method

```python
# Return dictionary matching TaskDetectionResponse schema
{
    'task_type': 'translate',  # str
    'confidence': 0.92,  # float (0.0-1.0)
    'reason': 'User explicitly requested translation',  # str
    'flow_type': 'general_chat',  # str (echoed from input)
    'candidates': [  # Optional, only for general_chat flow
        {
            'name': 'translate',
            'description': 'Translate text from one language to another',
            'certainty': 0.92
        }
    ]
}
```

---

## 10. Logging Strategy

### 10.1 Log Events

**Event 1: Task Detection Started**
```python
self.logger.info(
    "Task Detection Started",
    extra={
        'correlation_id': correlation_id,
        'flow_type': flow_type,
        'user_input_length': len(user_input),
        'user_input_preview': user_input[:100]
    }
)
```

**Event 2: Fixed Task Detected**
```python
self.logger.info(
    "Fixed Task Type Detected",
    extra={
        'flow_type': flow_type,
        'task_type': task_type,
        'confidence': 1.0,
        'processing_time_ms': processing_time
    }
)
```

**Event 3: Embedding Generated**
```python
self.logger.debug(
    "Embedding Generated for Task Search",
    extra={
        'embedding_dimension': len(embedding_vector),
        'generation_time_ms': generation_time
    }
)
```

**Event 4: Weaviate Search Complete**
```python
self.logger.info(
    "Task Similarity Search Complete",
    extra={
        'candidates_found': len(candidates),
        'top_certainty': candidates[0]['certainty'] if candidates else 0,
        'search_time_ms': search_time
    }
)
```

**Event 5: LLM Re-ranking Complete**
```python
self.logger.info(
    "LLM Re-ranking Complete",
    extra={
        'selected_task': task_type,
        'confidence': confidence,
        'reranking_time_ms': reranking_time
    }
)
```

**Event 6: Fallback Triggered**
```python
self.logger.warning(
    "Task Detection Fallback Triggered",
    extra={
        'reason': fallback_reason,
        'original_flow': flow_type,
        'fallback_task': 'general_chat'
    }
)
```

**Event 7: Task Detection Complete**
```python
self.logger.info(
    "Task Detection Complete",
    extra={
        'flow_type': flow_type,
        'task_type': task_type,
        'confidence': confidence,
        'total_time_ms': total_time,
        'used_fallback': is_fallback
    }
)
```

### 10.2 Error Logging

**Weaviate Connection Error**:
```python
self.logger.error(
    "Weaviate Connection Failed",
    extra={
        'error_type': type(e).__name__,
        'error_message': str(e),
        'retry_attempt': retry_count
    },
    exc_info=True  # Include stack trace
)
```

**LLM API Error**:
```python
self.logger.error(
    "LLM API Error During Task Detection",
    extra={
        'error_type': type(e).__name__,
        'error_message': str(e),
        'stage': 'embedding_generation' or 'reranking',
        'will_fallback': True
    },
    exc_info=True
)
```

---

## 11. Performance Considerations

### 11.1 Performance Targets

**Fixed Flow Detection**:
- Target: < 10ms
- No external API calls
- Simple dictionary lookup

**General Chat Detection (Full Pipeline)**:
- Target: 1.5-2.5 seconds
- Breakdown:
  - Embedding generation: 200-500ms
  - Weaviate search: 100-300ms
  - LLM re-ranking: 1000-1500ms

**Optimization Strategies**:

1. **Caching** (Future Enhancement):
   - Cache embeddings for common inputs
   - TTL: 1 hour
   - Key: hash of user_input
   - Saves 200-500ms per cache hit

2. **Batch Processing** (Future Enhancement):
   - If multiple requests in queue
   - Batch embedding generation
   - Process multiple searches in parallel

3. **Early Termination**:
   - If top candidate certainty > 0.95
   - Skip LLM re-ranking
   - Accept highest similarity directly
   - Saves ~1 second

4. **Parallel Execution**:
   - If implementing multiple re-rankers
   - Run them in parallel
   - Take consensus vote

### 11.2 Resource Usage

**Memory**:
- Embedding vector: ~1536 floats × 4 bytes = 6 KB
- Candidate list (5 tasks): ~2 KB
- LLM prompt: ~500 tokens = ~2 KB
- Total per request: ~10 KB

**API Calls**:
- 1 embedding generation call
- 1 Weaviate query
- 1 LLM re-ranking call
- Total: 3 external calls for general_chat

**Cost Estimation** (per 1000 requests):
- Embeddings: ~$0.001 (text-embedding-3-small)
- Re-ranking: ~$0.15 (gpt-4o-mini, ~100 tokens)
- Total: ~$0.151 per 1000 general_chat detections
- Fixed flow detections: $0 (no API calls)

---

## 12. Testing Strategy

### 12.1 Unit Tests

**File**: `tests/unit/test_task_decisioner.py`

**Test Cases**:

1. **test_detect_task_fixed_flow_file_analysis**
2. **test_detect_task_fixed_flow_image_generation**
3. **test_detect_task_general_chat_high_confidence**
4. **test_detect_task_general_chat_low_confidence**
5. **test_detect_task_no_candidates_found**
6. **test_embedding_generation_error**
7. **test_weaviate_connection_error**
8. **test_llm_reranking_invalid_response**
9. **test_confidence_threshold_boundary**
10. **test_invalid_flow_type**

### 12.2 Integration Tests

**File**: `tests/integration/test_task_decisioner_integration.py`

**Test Cases**:
1. **test_end_to_end_task_detection_with_real_weaviate**
2. **test_multiple_flows_sequential**
3. **test_concurrent_task_detection**

### 12.3 End-to-End Tests

**File**: `tests/e2e/test_orchestrator_with_task_decisioner.py`

---

## 13. Error Handling Matrix

| Error Scenario | Exception Type | Handling Strategy | Fallback | User Impact |
|---------------|----------------|-------------------|----------|-------------|
| Invalid flow_type | ValidationError | Log and raise | None | 400 Bad Request |
| Empty user_input | ValidationError | Log and raise | None | 400 Bad Request |
| Embedding API timeout | LLMError | Retry once, then fallback | general_chat | Response generated |
| Weaviate connection error | MemoryError | Log, fallback | general_chat | Response generated |
| No candidates found | (Not an error) | Log warning, fallback | general_chat | Response generated |
| Low LLM confidence | (Not an error) | Log info, fallback | general_chat | Response generated |
| LLM invalid JSON | (Not an error) | Log warning, use top candidate | Highest similarity | Response generated |
| Unexpected exception | OrchestrationError | Log with stack trace, raise | None | 500 Internal Error |

---

## 14. Example Scenarios

### Scenario 1: Translation Request (General Chat)

**Input**:
```python
user_input = "translate 'good morning' to French"
flow_type = "general_chat"
```

**Output**:
```python
{
    'task_type': 'translate',
    'confidence': 0.96,
    'reason': 'User explicitly requested translation to French',
    'flow_type': 'general_chat',
    'candidates': [...]
}
```

**Processing Time**: ~1.8 seconds

---

### Scenario 2: File Analysis Request (Fixed Flow)

**Input**:
```python
user_input = "summarize this document"
flow_type = "file_analysis"
```

**Output**:
```python
{
    'task_type': 'file_analysis',
    'confidence': 1.0,
    'reason': 'Fixed task for file analysis flow',
    'flow_type': 'file_analysis',
    'candidates': None
}
```

**Processing Time**: < 5ms

---

### Scenario 3: Ambiguous Input with Fallback

**Input**:
```python
user_input = "hmm, interesting"
flow_type = "general_chat"
```

**Output**:
```python
{
    'task_type': 'general_chat',
    'confidence': 1.0,
    'reason': 'Fallback due to low confidence (0.55)',
    'flow_type': 'general_chat',
    'candidates': [...]
}
```

**Processing Time**: ~1.9 seconds

---

### Scenario 4: Image Generation Request (Fixed Flow)

**Input**:
```python
user_input = "draw a sunset over mountains"
flow_type = "image_generation"
```

**Output**:
```python
{
    'task_type': 'image_generation',
    'confidence': 1.0,
    'reason': 'Fixed task for image generation flow',
    'flow_type': 'image_generation',
    'candidates': None
}
```

**Processing Time**: < 5ms

---

## 15. Summary Checklist

**Before Implementation**:
- [ ] Review PROJECT_PLAN_Version3.md architecture
- [ ] Review task_decisioner.xml diagram
- [ ] Understand flow → task dependency
- [ ] Set up Weaviate schema for TaskType
- [ ] Configure environment variables

**During Implementation**:
- [ ] Create app/services/task/decisioner.py with TaskDecisioner class
- [ ] Implement all 11 methods according to specifications
- [ ] Add proper error handling for each method
- [ ] Implement logging at key points
- [ ] Create app/schemas/task.py with Pydantic models
- [ ] Update app/utils/constants.py with enums
- [ ] Update app/config.py with task detection settings
- [ ] Integrate with Orchestrator in app/orchestrator.py

**After Implementation**:
- [ ] Write unit tests (10+ test cases)
- [ ] Write integration tests with real Weaviate
- [ ] Test all three flow types
- [ ] Test fallback scenarios
- [ ] Test error handling
- [ ] Verify performance targets met
- [ ] Document API in code comments

**Key Points to Remember**:
1. **Always receive flow_type as parameter** in detect_task method
2. **Fixed flows return immediately** without Weaviate search
3. **Only general_chat uses dynamic detection** with full pipeline
4. **Fallback is not failure** - it's graceful degradation
5. **Confidence threshold** determines fallback vs. acceptance
6. **LLM re-ranking** is the final decision maker
7. **Error handling** ensures system keeps running
8. **Logging** provides observability at every step

---

**Created**: 2025-11-06 14:06:51  
**Author**: HoMinhHao  
**Component**: Task Decisioner  
**Status**: Implementation Ready