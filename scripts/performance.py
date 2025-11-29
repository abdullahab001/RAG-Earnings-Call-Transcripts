# At the top of your file
import time
import tiktoken
from rouge_score import rouge_scorer
import earnings_call_transcript as ect
from earnings_call_transcript import *
# Pricing (per 1M tokens)
EMBEDDING_COST = 0.02
INPUT_COST = 0.150
OUTPUT_COST = 0.600

encoding = tiktoken.get_encoding("o200k_base")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def count_tokens(text):
    return len(encoding.encode(text))

# In your main loop, wrap your query:
start = time.time()

query = "What was the revenue?"
collection = ect.collection
openai_client = ect.openai_client
selected_company = "Ambarella"
conversation_history = []

# Your existing code
relevant_transcripts = query_transcripts(query, collection, openai_client, selected_company)
answer = generate_answers(query, relevant_transcripts['documents'], openai_client, conversation_history)

# Calculate metrics
latency = time.time() - start
query_tokens = count_tokens(query)
input_tokens = count_tokens("\n\n".join(relevant_transcripts['documents']) + query)
output_tokens = count_tokens(answer)

cost = (query_tokens/1_000_000 * EMBEDDING_COST) + (input_tokens/1_000_000 * INPUT_COST) + (output_tokens/1_000_000 * OUTPUT_COST)

# Quality (ROUGE-L with first retrieved chunk as reference)
quality = rouge.score(relevant_transcripts['documents'][0], answer)['rougeL'].fmeasure

# Print
print(f"\n ${cost:.6f} | ⏱ {latency:.2f}s | 🔢 {input_tokens+output_tokens:,} tokens |  Quality: {quality:.3f}")
print(f"\nAnswer: {answer}\n")