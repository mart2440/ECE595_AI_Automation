# ===========================
# Makefile — Purdue GenAI Automation
# Models: LLaMA 3.2 and Gemma 3:27b
# ===========================

# Python environment
PYTHON=./venv/bin/python

# Input & Output files
INPUT=prompts_experiment.csv

# =============================
# Run LLaMA (Purdue GenAI)
# =============================
run_llama:
	@echo ""
	@echo "Running prompts through LLaMA (llama3.2:latest)..."
	$(PYTHON) run_llm_truthfulqa.py \
		--input_csv $(INPUT) \
		--output_csv results_llama.csv \
		--base_url https://genai.rcac.purdue.edu/api/ \
		--model llama3.2:latest \
		--api_key_env GENAI_API_KEY

# =============================
# Run Gemma 27B (Purdue GenAI)
# =============================
run_gemma:
	@echo ""
	@echo "Running prompts through Gemma 3 (27B)..."
	$(PYTHON) run_llm_truthfulqa.py \
		--input_csv $(INPUT) \
		--output_csv results_gemma.csv \
		--base_url https://genai.rcac.purdue.edu/api/ \
		--model gemma3:27b \
		--api_key_env GENAI_API_KEY

# =============================
# Run ChatGPT (OpenAI)
# =============================
run_chatgpt:
	@echo ""
	@echo "Running prompts through ChatGPT..."
	$(PYTHON) run_llm_truthfulqa.py \
		--input_csv $(INPUT) \
		--output_csv results_chatgpt.csv \
		--base_url https://api.openai.com/v1/ \
		--model gpt-4.1 \
		--api_key_env OPENAI_API_KEY

