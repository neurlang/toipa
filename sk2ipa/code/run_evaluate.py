import transformers
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import optparse
import sys
import os.path
import csv


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--text', action='store', dest='text',
                      help='Text')
    parser.add_option('--orig', action='store', dest='orig',
                      help='Orig')
    parser.add_option('--old', action='store', dest='old',
                      help='Old')
    (params, _) = parser.parse_args(sys.argv)
    if not params.text:
        print("Speaker text is mandatory")
        sys.exit(-1)
    if not params.orig:
        print("Orig is mandatory")
        sys.exit(-1)
    if not params.old:
        print("Old is mandatory")
        sys.exit(-1)

    max_input_length = 128
    max_target_length = 128


    tokenizer = T5Tokenizer('m.model')

    model = AutoModelForSeq2SeqLM.from_pretrained("./my2.model", local_files_only=True)


    if os.path.isfile(params.text):

        dictionary = {}

        with open('dataset.txt', mode='r') as infile:
            for line in csv.reader(infile):
                l = line[0].split('\t')
                dictionary[l[0]] = l[1]

        with open(params.text,'r') as file:

            # reading each line    
            for line in file:

                printed=False

                # reading each word        
                for word in line.split():
   
                    # displaying the words           
                    origword = word.lower()
                    word = word.rstrip("!?.,")

                    ending = origword[len(word):]


                    try:
                        wrd = dictionary[word]
                        if params.old != "yes":
                            continue
                    except:
                        wrd = word

                        input_encodings1 = tokenizer.encode_plus(word,
                                         padding="max_length",
                                         max_length=max_input_length,
                                         return_tensors="pt")

                        gen1 = model.generate(input_ids=input_encodings1.input_ids,
                            attention_mask=input_encodings1.attention_mask,
                            max_length=max_target_length,
                            decoder_start_token_id=model.config.decoder_start_token_id
                            )

                        dec1 = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen1]

                        wrd = dec1[0]


                    if params.orig == "yes":
                        print(word, end ="\t")
                        print(wrd)
                    else:
                        printed=True
                        print(wrd + ending, end =" ")

                if printed:
                    print("")

    else:


        input_str1 = params.text.replace("_", " ").lower()

        input_encodings1 = tokenizer.encode_plus(input_str1,
                                         padding="max_length",
                                         max_length=max_input_length,
                                         return_tensors="pt")


        gen1 = model.generate(input_ids=input_encodings1.input_ids,
                      attention_mask=input_encodings1.attention_mask,
                      max_length=max_target_length,
                      decoder_start_token_id=model.config.decoder_start_token_id
                      )

        dec1 = [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen1]
        print(input_str1)
        print(dec1)

