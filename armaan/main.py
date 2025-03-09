from os import popen
from utils import arch_name_to_id
from generate_explanations import generate_explanations
from generate_scores import generate_scores
from generate_tokens import generate_tokens
from generate_acts import generate_acts

if __name__ == "__main__":
    # generate_explanations("LayernormSqueeze1eNeg4lr4eNeg4")
    # generate_explanations("2x4x4x2LayernormSqueeze2eNeg4lr4eNeg4")
    # generate_explanations("2x2LayernormSqueeze1eNeg4lr4eNeg4")

    generate_scores("LayernormSqueeze1eNeg4lr4eNeg4", "fuzz")
    generate_scores("2x4x4x2LayernormSqueeze2eNeg4lr4eNeg4", "fuzz")
    generate_scores("2x2LayernormSqueeze1eNeg4lr4eNeg4", "fuzz")


