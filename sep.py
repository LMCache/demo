import pdb


def _chunk_tokens(

            tokens,
            split_at
        ):

        #token_chunks = [list(g) for k, g in groupby(tokens, lambda x: x != split_at) if k]
        
        token_chunks = []  
        start_idx = 0  
        sub_start_idx = -1  
        for i, item in enumerate(tokens):  
            if sub_start_idx == -1 and item == split_at[0]:  
                sub_start_idx = i  
            elif sub_start_idx != -1 and item == split_at[-1]:  
                if i - sub_start_idx + 1 == len(split_at):  
                    token_chunks.append(tokens[start_idx:sub_start_idx])  
                    start_idx = i + 1
                    sub_start_idx = -1
            elif sub_start_idx != -1 and item != split_at[-1]:  
                pass  
            elif sub_start_idx == -1:  
                pass 
        if start_idx < len(tokens):  
            token_chunks.append(tokens[start_idx:])  
        
        tokens_new = []
        for token_chunk in token_chunks:
            tokens_new.extend(token_chunk)
        token_chunks = [token_chunk for token_chunk in token_chunks]
        return token_chunks
    
x = [1,422,422,41512]
pdb.set_trace()
print(11111)