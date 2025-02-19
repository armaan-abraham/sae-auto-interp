# %%

from nnsight import LanguageModel

from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from functools import partial

import torch
from IPython.display import HTML, display
import json

import ipywidgets as widgets
from IPython.display import clear_output

module_name = "layers.9.resid_pre"
width = 24576
arch = "2-2"
raw_dir = f"latents_{arch}"



# %%

def make_colorbar(min_value, max_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if(min_value < -negative_threshold):
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
    # Do positive
    if(max_value > positive_threshold):
        for i in range(1, num_colors+1):
            ratio = i / (num_colors)
            value = round((max_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    return colorbar

def value_to_color(activation, max_value, min_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    if activation > positive_threshold:
        ratio = activation/max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim()==2:
            array = array.tolist()
        else: 
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if isinstance(array[0], int):
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal"):
    # text_spacing = "0.07em"
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok] for tok in toks]
    print(len(activations))
    print(len(toks))
    highlighted_text = []
    # Make background black
    # highlighted_text.append('<body style="background-color:black; color: white;">')
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    max_value = max([max(activ) for activ in activations])
    min_value = min([min(activ) for activ in activations])
    if(logit_diffs is not None and model_type != "reward_model"):
        logit_max_value = max([max(activ) for activ in logit_diffs])
        logit_min_value = min([min(activ) for activ in logit_diffs])

    # Add color bar
    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
        for act_ind, (a, t) in enumerate(zip(act, tok)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp").replace("<bos>","BOS")}</span>')
            if(logit_diffs is not None and model_type != "reward_model"):
                logit_diffs_act = logit_diffs[seq_ind][act_ind]
                _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
        if(logit_diffs is not None and model_type=="reward_model"):
            reward_change = logit_diffs[seq_ind].item()
            text_color, background_color = value_to_color(reward_change, 10, -10)
            highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
        # highlighted_text.append('<br><br>')
    # highlighted_text.append('</body>')
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text


# %%
model = LanguageModel("gpt2", device_map="cpu", dispatch=True)
model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# %%
def load_examples():
    
    feature_cfg = FeatureConfig(width=width)
    experiment_cfg = ExperimentConfig(n_random=0,train_type="quantiles",n_examples_train=50,example_ctx_len=64)

    #module = f".model.layers.{layer_name}.post_feedforward_layernorm"
    module = module_name

    print(f"Raw dir: {raw_dir}")
    tokens = torch.load("/root/sae-auto-interp/examples/armaan/tokens.pt")
    
    dataset = FeatureDataset(
        raw_dir=raw_dir,
        cfg=feature_cfg,
        modules=[module],
        features={module:torch.tensor(torch.arange(0, 10))},
        tokenizer=model.tokenizer,
        tokens=tokens,
    )
    constructor=partial(
                default_constructor,
                tokens=dataset.tokens,
                n_random=experiment_cfg.n_random, 
                ctx_len=64, 
                max_examples=10000
            )
    sampler = partial(sample,cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

    all_examples = {}
    maximum_activations = {}
    for record in loader:
        train_examples = record.train
        all_examples[str(record.feature)] = train_examples
        maximum_activations[str(record.feature)] = record.max_activation

    return all_examples, maximum_activations

def plot_examples():
    all_examples, maximum_activations = load_examples()
    keys = list(all_examples.keys())
    
    current_index = [0]  # Use a list to store the current index so it can be modified in the callback
    explanations = {}  # Dictionary to store explanations

    def display_example(index):
        key = keys[index]
        print(key)
        list_tokens = []
        list_activations = []
        for example in all_examples[key]:
            example_tokens = example.tokens
            activations = example.activations / maximum_activations[key]
            list_tokens.append(example_tokens)
            list_activations.append(activations.tolist())

        display(HTML(tokens_and_activations_to_html(list_tokens, list_activations, model.tokenizer)))

    def on_submit(b):
        key = keys[current_index[0]]
        explanations[key] = text_box.value
        current_index[0] = (current_index[0] + 1) % len(keys)
        clear_output(wait=True)
        display(widgets.HBox([text_box, submit_button, skip_button, save_button]))
        display_example(current_index[0])
    
    def on_skip(b):
        current_index[0] = (current_index[0] + 1) % len(keys)
        clear_output(wait=True)
        display(widgets.HBox([text_box, submit_button, skip_button, save_button]))
        display_example(current_index[0])
        
    def on_save(b):
        with open(f"{module_name}_explanations.json", "w") as f:
            json.dump(explanations, f)
        print(f"Explanations saved to {module_name}_explanations.json")

    text_box = widgets.Text(description="Explanation:")
    submit_button = widgets.Button(description="Submit")
    submit_button.on_click(on_submit)
    skip_button = widgets.Button(description="Skip")
    skip_button.on_click(on_skip)
    save_button = widgets.Button(description="Save")
    save_button.on_click(on_save)
    display(widgets.HBox([text_box, submit_button, skip_button, save_button]))
    display_example(current_index[0])

# %%

def make_colorbar(min_value, max_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if(min_value < -negative_threshold):
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
    # Do positive
    if(max_value > positive_threshold):
        for i in range(1, num_colors+1):
            ratio = i / (num_colors)
            value = round((max_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    return colorbar

def value_to_color(activation, max_value, min_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    if activation > positive_threshold:
        ratio = activation/max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim()==2:
            array = array.tolist()
        else: 
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if isinstance(array[0], int):
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal"):
    # text_spacing = "0.07em"
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok] for tok in toks]
    print(len(activations))
    print(len(toks))
    highlighted_text = []
    # Make background black
    # highlighted_text.append('<body style="background-color:black; color: white;">')
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    max_value = max([max(activ) for activ in activations])
    min_value = min([min(activ) for activ in activations])
    if(logit_diffs is not None and model_type != "reward_model"):
        logit_max_value = max([max(activ) for activ in logit_diffs])
        logit_min_value = min([min(activ) for activ in logit_diffs])

    # Add color bar
    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
        for act_ind, (a, t) in enumerate(zip(act, tok)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp").replace("<bos>","BOS")}</span>')
            if(logit_diffs is not None and model_type != "reward_model"):
                logit_diffs_act = logit_diffs[seq_ind][act_ind]
                _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
        if(logit_diffs is not None and model_type=="reward_model"):
            reward_change = logit_diffs[seq_ind].item()
            text_color, background_color = value_to_color(reward_change, 10, -10)
            highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
        # highlighted_text.append('<br><br>')
    # highlighted_text.append('</body>')
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text


# %%
plot_examples()

# %%
