from utils import arch_name_to_id
from generate_explanations import generate_explanations
from generate_scores import generate_scores

if __name__ == "__main__":
    for arch in arch_name_to_id.keys():
        generate_explanations(arch)
        generate_scores(arch)

