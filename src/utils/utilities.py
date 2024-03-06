import tiktoken
def count_num_tokens(text:str,model:str)->int:
    """
    This method is responsible for counting the number of tokens in a given text.
    It uses the tokenization method from the model to count the number of tokens in the text.
    Args:
        text (str): The input text
        model (str, optional): The name of the GPT model to use. Defaults to the model specified in the app config.
    Returns:
        int: The number of tokens in the text
    """
    encoding = tiktoken.encoding_for_model(model)
    #tokenizer = tiktoken.get_tokenizer(model)
    #tokens = tokenizer.tokenize(text)
    return len(encoding.encode(text))
    