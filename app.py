import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

token = os.getenv('HF_TOKEN')

# token = token # hugging face token
@st.cache_resource
def load_model(base_model_path) :
    """
    Load the base model and apply the adapter.
    """

    print('START OF THE APP')
    # Load the base model and tokenizer
    #token = token 
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', token=token) # meta-llama/Llama-3.2-1B
    base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', token=token,device_map="auto", low_cpu_mem_usage=True,trust_remote_code=True,torch_dtype=torch.float16)
    print('Loaded the BASE MODEL AND TOKENIZER ')
    print(f"Base Model Path: {base_model_path}")
    print(f"Adapter Path: {adapter_path}")
    # Load the adapter
    model =  PeftModel.from_pretrained(base_model,'eromanova115/CyberSecurityAIAssistant',token=token)
#    adapter_config_path = os.path.dirname('CyberSecurityAssistant/adapter_config.json') 
    # print(f"Adapter Config Path: {adapter_config_path}")
    # print('type of adapter config path ',type(adapter_config_path))
    # model = PeftModel.from_pretrained(
    #     base_model,
    #     adapter_path,
    #     config=adapter_config_path,
    #     torch_dtype='auto'
    # )
#    model = PeftModel.from_pretrained(base_model,adapter_path)
    model = model.merge_and_unload()
    print('Model is merged successful')
    return model, tokenizer


# Streamlit UI
st.title("Cybersecurity AI ASSISTANT LLM Security")


# Sidebar inputs for model paths
base_model_path = st.sidebar.text_input("Base Model Path from HF", 'meta-llama/Llama-3.2-3B')
adapter_path = st.sidebar.text_input("Adapter Safetensors Path", 'CyberSecurityAssistant')
adapter_config_path = st.sidebar.text_input("Adapter Config Path", 'CyberSecurityAssistant/adapter_config.json') # CyberSecurityAssistant\adapter_config.json
print(f"{base_model_path=}")

# Temperature slider
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)

# Load the model
if base_model_path and adapter_path and adapter_config_path:
    try:
        with st.spinner("Loading model..."):
            model, tokenizer = load_model(base_model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        model, tokenizer = None, None
else:
    st.warning("Please provide paths to the model and adapter files in the sidebar.")


# SYSTEM PROMPT

# GLOBAL VARIABLE INSTRUCTION
instruction= 'You are a Cybersecurity AI Assistant, will be glad to answer your questions related to Cybersecurity, particularly LLM Security.'


# Chat Interface
if model and tokenizer:
    user_input = st.text_input("Your message", "")
    user_input= f'{instruction} \n\nUser: {user_input}\nAI'
    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Tokenize input
                input_ids = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
                # Generate response
                outputs = model.generate(input_ids, max_new_tokens=512, temperature=temperature)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                st.write(f"**Response:** {response}")
            except Exception as e:
                st.error(f"Error generating response: {e}")


# streamlit run app.py