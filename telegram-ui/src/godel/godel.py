from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

def generate(input):
    dialog = []
    dialog.append(input)
    # Instruction for a chitchat task
    instruction = f'Instruction: given a dialog context, you need to response empathically.'    
    knowledge = ''
    # if knowledge != '':
    #     knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# # sample dialog
# dialog = []
# while True:
#     chat = input('Me: ')
#     dialog.append(chat)
#     response = generate(dialog)
#     print("GODEL: ",response)
# while True:
#     chat = input('Me: ')
#     response = generate(chat)
#     print("GODEL: ",response)


