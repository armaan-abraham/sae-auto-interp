from utils import arch_name_to_id
from generate_explanations import generate_explanations
from generate_scores import generate_scores
from generate_tokens import generate_tokens
from generate_acts import generate_acts

if __name__ == "__main__":
    generate_explanations("2-4-4-2")
    generate_scores("2-4-4-2", "detection")
    generate_scores("2-4-4-2", "fuzz")

