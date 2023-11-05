import numpy as np
from tqdm import tqdm

from encoder import get_encoder
from tools import get_params


# Multi-head attention

def softmax(x):
    numerator = np.exp(x - np.max(x, axis=1, keepdims=True))
    denominator = np.sum(numerator, axis=1, keepdims=True)
    softmax_values = numerator / denominator
    
    return softmax_values


def attention(Q, K, V):
    product = (Q @ K.T) / np.sqrt(Q.shape[1])
    attention_weights = softmax(product)

    return (attention_weights @ V)


def masked_attention(Q, K, V, mask):
    product = (Q @ K.T) / np.sqrt(Q.shape[1])
    masked_product = product + mask
    attention_weights = softmax(masked_product)
    
    return (attention_weights @ V)


def linear_projection(x, w, b):
    pass


def multi_head_attention(x, attn, number_of_heads):
    w_1, b_1 = attn["c_attn"]["w"], attn["c_attn"]["b"]
    w_2, b_2 = attn["c_proj"]["w"], attn["c_proj"]["b"]
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    """
        Your code here
    """
    return x


# Transformer blocks and GPT2


def gelu(x):
    pass


def layer_normalization(x, g, b, eps=1e-5):
    pass


def feed_forward_network(x, mlp):
    w_1, b_1 = mlp["c_fc"]["w"], mlp["c_fc"]["b"]
    w_2, b_2 = mlp["c_proj"]["w"], mlp["c_proj"]["b"]
    """
        Your code here
    """
    return x


def transformer_block(x, block, number_of_heads):
    mlp, attn = block["mlp"], block["attn"]
    ln_1, ln_2 = block["ln_1"], block["ln_2"]
    g_1, b_1, g_2, b_2 = ln_1["g"], ln_1["b"], ln_2["g"], ln_2["b"]
    """
        Your code here
    """
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, number_of_heads):
    g_final, b_final = ln_f["g"], ln_f["b"]
    x = wte[inputs] + wpe[range(len(inputs))]  # Step 1: Sum positional encoding and token encoding 
    """
        Your code here
    """
    return x @ wte.T


def generate(input_text, tokens_to_generate=40, model_size="124M", models_dir="models", loading_bar=True):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    
    hparams, params = get_params(model_size, models_dir)
    encoder = get_encoder(model_size, models_dir)
    number_of_heads = hparams["n_head"]
    max_context = hparams["n_ctx"]

    # Port the input text to ids
    input_ids = encoder.encode(input_text)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + tokens_to_generate < max_context

    # generate output ids
    output_ids = []
    
    if loading_bar:
        loop_range = tqdm(range(tokens_to_generate), "Thinking...")
    else:
        loop_range = range(tokens_to_generate)

    for _ in loop_range:
        # Call our gtp2 model with input plus generated tokens
        output = gpt2(input_ids + output_ids, **params, number_of_heads=number_of_heads) 

        # Get the next token from the output
        next_id = np.argmax(output[-1])

        # Save the result
        output_ids.append(int(next_id))

    # Port the output ids to text
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    
    Test your implemetntation with something like this:
    print(generate("Hello! How do you do?"))

    You can try out different sized models from this list: ["124M", "355M", "774M", "1558M"]
    Make sure you have enough space on your device since the bigger models are quite large.
    """
    
    print("[+]Part 1.1")
    print(softmax(np.array([[-1., 0.], [0.2, 1.]])))
    
    print("\n[+]Part 1.2")
    np.random.seed(4321)
    q = np.random.rand(3,2)
    k = np.random.rand(3,2)
    v = np.random.rand(3,2)
    x = attention(q, k, v)
    print(x)
    
    print("\n[+]Part 1.3")
    np.random.seed(4321)
    nf = 10
    q = np.random.rand(nf,2)
    k = np.random.rand(nf,2)
    v = np.random.rand(nf,2)
    mask = (1 - np.tri(nf)) * -1e10
    x = masked_attention(q, k, v, mask)
    print(x)