import torch
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
def optimized_generation(
        reward_model, model, tokenizer, device,
        question, input_text, original_answer, 
        original_hidden_states_list, input_ids, start_index=0, 
        max_num_steps=10, lr=0.03, max_new_tokens=1024,
        grad_clip=None, k=0.1, reward_threshold=-0.2, prompt_length=None):
    '''
    Generate answer using optimized generation process

    Args:
        reward_model: reward model
        model: language model
        tokenizer: tokenizer
        device: device to use
        question: question
        input_text: formatted prompt
        original_answer: original generated answer
        original_hidden_states_list: list of hidden states for each token
        input_ids: the input_ids of original generation
        start_index: the start index of the optimized hidden states
        max_num_steps: number of optimization steps
        lr: learning rate
        max_new_tokens: maximum number of new tokens to generate
        grad_clip: gradient clipping threshold
        k: ratio of update length to the total length of hidden states
        reward_threshold: threshold for the reward to stop optimization
        
    Returns:
        final_answer: the final generated answer
        reward_history: list of rewards during optimization
        original_length: length of the original answer
        optimized_length: length of the optimized answer
        update_length: length of the optimized hidden states
    '''
    eos_token = tokenizer.eos_token
    stop_words.append(eos_token)
    reward_history = []
    initial_reward = reward_model.get_reward(question, original_answer)
    
    print(f"-- Original Output: {original_answer} -- Initial Reward: {initial_reward}")
    reward_history.append(initial_reward)
    current_reward = initial_reward
    
    original_length = len(original_hidden_states_list)
    optimized_length = 0
    
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    base_input_ids = inputs.input_ids.clone()
    
    # Calculate prompt length if not provided
    if prompt_length is None:
        prompt_length = len(base_input_ids[0])
    
    # Calculate the number of generated tokens (excluding prompt)
    generated_length = original_length - prompt_length
    
    # grab update fraction from generated tokens only
    update_length = min(int(k * generated_length), 300)
    if update_length <= 0:
        print("Update Length Zero!!!")
        final_answer = original_answer
        return final_answer, reward_history, original_length, optimized_length, update_length

    # Index from the generated part (after prompt)
    actual_start = prompt_length + start_index
    actual_end = min(actual_start + update_length, len(original_hidden_states_list))
    
    optimized_hidden_states = torch.nn.Parameter(torch.stack(
        [state.clone().detach().requires_grad_(True)
        for state in original_hidden_states_list[actual_start:actual_end]])
    )
    
    # configure optimizer
    optimizer = torch.optim.Adam([optimized_hidden_states], lr=lr)
    
    # Start with just the prompt - no need for original_seq
    base_prompt_ids = base_input_ids.clone()
    new_answer = None
    
    # optimization loop
    for step in range(max_num_steps):
        # Always start fresh from the prompt
        input_ids = base_prompt_ids.clone()
        if current_reward > reward_threshold:
            final_answer = new_answer if new_answer is not None else original_answer
            optimized_length = len(tokenizer.encode(final_answer))
            print(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
            return final_answer, reward_history, original_length, optimized_length, update_length
        
        optimizer.zero_grad()
        
        # Ensure correct dimensions for hidden states
        if optimized_hidden_states.dim() == 2:
            hidden_for_lm = optimized_hidden_states.unsqueeze(1)  # Add sequence dimension
        else:
            hidden_for_lm = optimized_hidden_states
            
        logits = model.lm_head(hidden_for_lm)  # [update_length, 1, vocab_size]
        probs = torch.softmax(logits, dim=-1) + 1e-8
        
        # Handle different dimension cases
        if logits.dim() == 3:
            next_token_ids = torch.argmax(probs, dim=-1)  # [update_length, 1]
            if next_token_ids.dim() == 2 and next_token_ids.shape[1] == 1:
                next_token_ids = next_token_ids.squeeze(-1)  # [update_length]
            log_pi_xz = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
        else:
            next_token_ids = torch.argmax(probs, dim=-1)
            log_pi_xz = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
        
        # total loss
        loss = - current_reward * log_pi_xz.sum()
        print(f"Step {step + 1}: Loss = {loss.item():.3f}")
        loss.backward(retain_graph=True)
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], grad_clip)
        optimizer.step()
        
        # Generate tokens from optimized hidden states
        with torch.no_grad():
            if optimized_hidden_states.dim() == 2:
                hidden_for_lm = optimized_hidden_states.unsqueeze(1)
            else:
                hidden_for_lm = optimized_hidden_states
                
            next_tokens = torch.argmax(model.lm_head(hidden_for_lm), dim=-1)
            
            # Handle dimensions
            if next_tokens.dim() == 2 and next_tokens.shape[1] == 1:
                next_tokens = next_tokens.squeeze(-1)
            
            # Create new input sequence: prompt + optimized tokens
            next_tokens_tensor = next_tokens.unsqueeze(0) if next_tokens.dim() == 1 else next_tokens
            input_ids = torch.cat([input_ids, next_tokens_tensor], dim=-1)
                
        # Generate the rest of the answer
        generated_ids = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Check sequence length to avoid exceeding model's max
                if input_ids.shape[1] >= 4096:
                    print(f"Warning: Sequence length {input_ids.shape[1]} approaching limit, stopping generation")
                    break
                    
                outputs = model.model(input_ids, output_hidden_states=True)
                hidden_states = outputs[0][:, -1]
                logits = model.lm_head(hidden_states)
                next_token_id = torch.argmax(logits, dim=-1)
                
                # Check for EOS
                token_value = next_token_id.item() if next_token_id.dim() == 0 else next_token_id[0].item()
                if token_value == tokenizer.eos_token_id:
                    break
                    
                generated_ids.append(token_value)
                
                # Add to sequence
                if next_token_id.dim() == 0:
                    next_token_id = next_token_id.unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
        
        # Decode the complete sequence
        full_ids = input_ids[0].tolist()
        new_answer = tokenizer.decode(full_ids[prompt_length:], skip_special_tokens=True)
        
        del outputs, hidden_states, logits, input_ids
        torch.cuda.empty_cache()
        current_reward = reward_model.get_reward(question, new_answer)
        print(f"-- New Answer: {new_answer}, -- Current Reward: {current_reward}")
            
        reward_history.append(current_reward)
        
    final_answer = new_answer
    optimized_length = len(tokenizer.encode(final_answer))
    print(f"-- Final answer: {final_answer}")
    return final_answer, reward_history, original_length, optimized_length, update_length

