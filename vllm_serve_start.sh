%%shell
#!/bin/bash

# Vllm parameters
MODEL="Qwen/Qwen3-8B-AWQ"
PORT=8000
MAX_MODEL_LEN=8192
MAX_BATCHED_TOKENS=8192
GPU_MEM_UTIL=0.85
LOG_FILE="vllm_server.log"

# start server
echo "Starting vLLM server with model: $MODEL on port: $PORT..."

> "$LOG_FILE"

nohup vllm serve "$MODEL" \
    --enable-prefix-caching \
    --trust-remote-code \
    --dtype half \
    --reasoning-parser deepseek_r1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT" > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo "vLLM process started in background with PID: $VLLM_PID"
echo "Waiting for 'Application startup complete.' in $LOG_FILE..."

# Startup check
while true; do
    if grep -q "Application startup complete." "$LOG_FILE"; then
        echo -e "\n[SUCCESS] vLLM server is up and ready to accept requests!"
        break
    fi
    
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo -e "\n[ERROR] vLLM process ($VLLM_PID) died before startup completed."
        echo "Check the last few lines of $LOG_FILE for the error:"
        tail -n 15 "$LOG_FILE"
        exit 1
    fi

    echo -n "."
    sleep 5
done