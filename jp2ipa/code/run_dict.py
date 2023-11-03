import transformers
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import optparse
import sys
import os.path
import csv
import torch
from tqdm import tqdm

def rmlast(word):
    if len(word) > 0:
        return word.rstrip(word[-1])
    return word

def batch(iterable, n = 1):
   current_batch = []
   for item in iterable:
       current_batch.append(item)
       if len(current_batch) == n:
           yield current_batch
           current_batch = []
   if current_batch:
       yield current_batch

def use(wordbuf):
    input_encodings1 = tokenizer(wordbuf,
                                 padding="max_length",
                                 max_length=max_input_length,
                                 return_tensors="pt")

    input_encodings1 = input_encodings1.to(device)

    gen1 = model.generate(input_ids=input_encodings1.input_ids,
                    attention_mask=input_encodings1.attention_mask,
                    max_length=max_target_length,
                    decoder_start_token_id=model.config.decoder_start_token_id
                    )

    dec1 = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen1]

    for i in range(len(dec1)):
        print(wordbuf[i], end ="\t")
        print(dec1[i])

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--text', action='store', dest='text',
                      help='Text')
    parser.add_option('--batchsize', action='store', dest='batchsize',
                      help='BatchSize')
                      
    (params, _) = parser.parse_args(sys.argv)
    if not params.text:
        print("Speaker text is mandatory")
        sys.exit(-1)

    #512 74
    #256 70
    #128 68
    #96 67
    #64 66
    #16 37
    if not params.batchsize:
        batch_size=96
    else:
        batch_size=int(params.batchsize)

    max_input_length = 128
    max_target_length = 128


    tokenizer = T5Tokenizer('m.model')
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    model = AutoModelForSeq2SeqLM.from_pretrained("./my2.model", local_files_only=True)
    model = model.to(device)

    if os.path.isfile(params.text):

        dictionary = {}

        with open('dataset.txt', mode='r') as infile:
            for line in csv.reader(infile):
                l = line[0].split('\t')
                dictionary[l[0]] = l[1]

        count_of_lines_in_any_textFile = sum(1 for l in open(params.text,'r'))

        with open(params.text,'r') as file:

            wordbuf = []

            # reading each line    
            for word in batch(tqdm(file, total=count_of_lines_in_any_textFile), batch_size - len(wordbuf)):

                for i in range(len(word)):
                    word[i] = word[i].lower().rstrip("!?.,\n\r")

                for i in range(len(word)):
                    while i < len(word):
                        try:
                            wrd = dictionary[word[i]]
                            word.pop(i)
                            continue
                        except:
                            break

                wordbuf.extend(word)

                if len(wordbuf) < batch_size:
                    continue

                use(wordbuf)

                wordbuf = []

            use(wordbuf)

    else:


        input_str1 = params.text.replace("_", " ").lower()

        input_encodings1 = tokenizer.encode_plus(input_str1,
                                         padding="max_length",
                                         max_length=max_input_length,
                                         return_tensors="pt")

        input_encodings1 = input_encodings1.to(device)

        gen1 = model.generate(input_ids=input_encodings1.input_ids,
                      attention_mask=input_encodings1.attention_mask,
                      max_length=max_target_length,
                      decoder_start_token_id=model.config.decoder_start_token_id
                      )

        dec1 = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen1]
        print(input_str1)
        print(dec1)

