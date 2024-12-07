import pandas as pd
import os

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC,  Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
import torchaudio

import random

import statistics

import jiwer

import kagglehub

import re

import matplotlib.pyplot as plt

def get_part_before_number(s):
    match = re.match(r"^[^\d]*", s)
    return match.group(0) if match else ""

def list_mp3_files(directory):
    return [file for file in os.listdir(directory) if file.endswith('.mp3')]

def bar_graph_class_count(data_dict):
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    plt.bar(keys, values, color='skyblue')

    plt.xlabel('Languages')
    plt.ylabel('Values')
    plt.title('Bar Graph of Dictionary Data')

    # Rotate the x-axis labels for better readability (if needed)
    plt.xticks(rotation=45)

    # Display the bar graph
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def get_data():
    # Download latest version
    print('downloading data')
    path = kagglehub.dataset_download("rtatman/speech-accent-archive")

    print("Path to dataset files:", path) # /Users/maduryasuresh/.cache/kagglehub/datasets/rtatman/speech-accent-archive/versions/2

    return

# Load and resample audio
def preprocess_audio(file_path):
    speech_array, sampling_rate = torchaudio.load(file_path)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(speech_array)
    return speech_array.squeeze()

def get_wer(inferred_text):
    # Reference and hypothesis (ASR output)
    reference = ("please call stella ask her to bring these things with her from the store six spoons of fresh snow peas five thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big toy frog for the kids she can scoop these things into three red bags and we will go meet her wednesday at the train station").upper() # ground truth
    #print(reference)
    # Calculate WER
    wer = jiwer.wer(reference, inferred_text)
    #print("Word Error Rate:", wer)

    return wer

def infer_csv(audio_files):
    # Load pre-trained model and processor
    model_name = "facebook/wav2vec2-base-960h"  # Change to your preferred model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    language_listOfWER = {} # language --> list of WER gotten
    
    for (ind, audio_file) in enumerate(audio_files):
        if ind % 50 == 0:
            print('file num:', ind)
        print(audio_file)
        audio_file = "./speech_accent_archive/recordings/" + audio_file
        speech = preprocess_audio(audio_file)

        # Process the audio for input
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        #print("Transcription:", transcription)
        wer = get_wer(transcription)

        after_slash_position = (audio_file.rfind('/')) + 1
        lang = audio_file[after_slash_position:]
        lang = get_part_before_number(lang)
        print('lang:', lang)
        print('wer:', wer)

        if lang not in language_listOfWER:
            temp = []
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)
        else:
            temp = language_listOfWER[lang]
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)

        #if ind == 10:
        #    break # test
        
        
    print(language_listOfWER)

    langs_list = list(language_listOfWER.keys())
    listWER_list = list(language_listOfWER.values())
    avgWER_list = []
    stdWER_list = []

    for (index, lang) in enumerate(langs_list):
        print(lang, listWER_list[index])
        mean = float(statistics.mean(listWER_list[index]))
        std_dev = 0
        if len(listWER_list[index]) != 1:
            std_dev = float(statistics.stdev(listWER_list[index]))
        avgWER_list.append(mean)
        stdWER_list.append(std_dev)

    print(avgWER_list, stdWER_list)

    df = pd.DataFrame({'Language': langs_list, 'List of WERs': listWER_list, 'Mean WER': avgWER_list, 'Std Dev WER': stdWER_list})

    print(df.head)

    df.to_csv('Language_WER.csv', index=False)


    return

def infer_csv(audio_files):
    # Load pre-trained model and processor
    model_name = "facebook/wav2vec2-base-960h"  # Change to your preferred model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    language_listOfWER = {} # language --> list of WER gotten
    
    for (ind, audio_file) in enumerate(audio_files):
        if ind % 50 == 0:
            print('file num:', ind)
        print(audio_file)
        audio_file = "./speech_accent_archive/recordings/" + audio_file
        speech = preprocess_audio(audio_file)

        # Process the audio for input
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        #print("Transcription:", transcription)
        wer = get_wer(transcription)

        after_slash_position = (audio_file.rfind('/')) + 1
        lang = audio_file[after_slash_position:]
        lang = get_part_before_number(lang)
        print('lang:', lang)
        print('wer:', wer)

        if lang not in language_listOfWER:
            temp = []
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)
        else:
            temp = language_listOfWER[lang]
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)

        #if ind == 10:
        #    break # test
        
        
    print(language_listOfWER)

    langs_list = list(language_listOfWER.keys())
    listWER_list = list(language_listOfWER.values())
    avgWER_list = []
    stdWER_list = []

    for (index, lang) in enumerate(langs_list):
        print(lang, listWER_list[index])
        mean = float(statistics.mean(listWER_list[index]))
        std_dev = 0
        if len(listWER_list[index]) != 1:
            std_dev = float(statistics.stdev(listWER_list[index]))
        avgWER_list.append(mean)
        stdWER_list.append(std_dev)

    print(avgWER_list, stdWER_list)

    df = pd.DataFrame({'Language': langs_list, 'List of WERs': listWER_list, 'Mean WER': avgWER_list, 'Std Dev WER': stdWER_list})

    print(df.head)

    df.to_csv('Language_WER.csv', index=False)


    return


def infer_checkpoint_csv(audio_files):
    # Load pre-trained model and processor
    model = Wav2Vec2ForCTC.from_pretrained("./checkpoint-2500")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    language_listOfWER = {} # language --> list of WER gotten
    
    for (ind, audio_file) in enumerate(audio_files):
        if ind % 50 == 0:
            print('file num:', ind)
        print(audio_file)
        audio_file = "./speech_accent_archive/recordings/" + audio_file
        speech = preprocess_audio(audio_file)

        # Process the audio for input
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = (processor.batch_decode(predicted_ids)[0]).upper()
        print("Transcription:", transcription)
        wer = get_wer(transcription)

        after_slash_position = (audio_file.rfind('/')) + 1
        lang = audio_file[after_slash_position:]
        lang = get_part_before_number(lang)
        print('lang:', lang)
        print('wer:', wer)

        if lang not in language_listOfWER:
            temp = []
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)
        else:
            temp = language_listOfWER[lang]
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)

        #if ind == 10:
        #    break # test
        
        
    print(language_listOfWER)

    langs_list = list(language_listOfWER.keys())
    listWER_list = list(language_listOfWER.values())
    avgWER_list = []
    stdWER_list = []

    for (index, lang) in enumerate(langs_list):
        print(lang, listWER_list[index])
        mean = float(statistics.mean(listWER_list[index]))
        std_dev = 0
        if len(listWER_list[index]) != 1:
            std_dev = float(statistics.stdev(listWER_list[index]))
        avgWER_list.append(mean)
        stdWER_list.append(std_dev)

    print(avgWER_list, stdWER_list)

    df = pd.DataFrame({'Language': langs_list, 'Mean WER': avgWER_list, 'Std Dev WER': stdWER_list})

    print(df.head)

    df.to_csv('Language_WER_Checkpoint.csv', index=False)


    return



def gather_class_stats(files):
    language_count = {}
    for file in files:
        language = get_part_before_number(file)
        if language not in language_count:
            language_count[language] = 1
        else:
            language_count[language] = language_count[language] + 1

    language_count = dict(sorted(language_count.items(), key=lambda item: item[1], reverse=True))
    filtered_language_count = {key: value for key, value in language_count.items() if value >= 12} # just keep some for display

    #print(language_count)

    return language_count, filtered_language_count

def bar_graph_class_count(data_dict):
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    plt.bar(keys, values, color='skyblue')

    plt.xlabel('Languages')
    plt.ylabel('Counts')
    plt.title('Languages Represented in Speech Accent Archive Dataset')

    # Rotate the x-axis labels for better readability (if needed)
    plt.xticks(rotation=45)

    # Display the bar graph
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def sort(df):
    languages = df['Language'].to_list()
    means = df['Mean WER'].to_list()
    std = df['Std Dev WER'].to_list()

    zipped_sorted = sorted(zip(languages, means, std), key=lambda x: x[1])
    sorted_languages, sorted_means, sorted_std = zip(*zipped_sorted)

    df_new = pd.DataFrame({'Language': sorted_languages, 'Mean WER': sorted_means, 'Std Dev': sorted_std})

    df_new.to_csv('Language_WER_Checkpoint_Sorted.csv', index=False)

    return

def infer_specific_checkpoint(audio_files): # just pass one audio file
    # Load pre-trained model and processor
    model = Wav2Vec2ForCTC.from_pretrained("./checkpoint-2500")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    language_listOfWER = {} # language --> list of WER gotten
    
    for (ind, audio_file) in enumerate(audio_files):
        if ind % 50 == 0:
            print('file num:', ind)
        print(audio_file)
        audio_file = "./speech_accent_archive/recordings/" + audio_file
        speech = preprocess_audio(audio_file)

        # Process the audio for input
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = (processor.batch_decode(predicted_ids)[0]).upper()
        print("Transcription:", transcription)
        wer = get_wer(transcription)

        after_slash_position = (audio_file.rfind('/')) + 1
        lang = audio_file[after_slash_position:]
        lang = get_part_before_number(lang)
        print('lang:', lang)
        print('wer:', wer)

        if lang not in language_listOfWER:
            temp = []
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)
        else:
            temp = language_listOfWER[lang]
            temp.append(wer)
            language_listOfWER[lang] = temp
            print('temp:', temp)

        #if ind == 10:
        #    break # test
        
        
    print(language_listOfWER)

    langs_list = list(language_listOfWER.keys())
    listWER_list = list(language_listOfWER.values())
    avgWER_list = []
    stdWER_list = []

    for (index, lang) in enumerate(langs_list):
        print(lang, listWER_list[index])
        mean = float(statistics.mean(listWER_list[index]))
        std_dev = 0
        if len(listWER_list[index]) != 1:
            std_dev = float(statistics.stdev(listWER_list[index]))
        avgWER_list.append(mean)
        stdWER_list.append(std_dev)

    print(avgWER_list, stdWER_list)

    df = pd.DataFrame({'Language': langs_list, 'Mean WER': avgWER_list, 'Std Dev WER': stdWER_list})

    print(df.head)

    df.to_csv('Language_WER_Checkpoint.csv', index=False)


    return

def display_WER(df):
    # Step 1: Select rows based on multiple values in column "A"
    selected_values = ["irish", "english", "afrikaans", "slovenian", "tagalog", "croatian", "hindi", "kiswahili", "vietnamese", "lao", "quechua"]  # Specify the desired values
    langs = df["Language"].to_list()

    #infer_test(['irish1.mp3'])
    #return

    filtered_df = df[df["Language"].isin(selected_values)]

    # Step 2: Display corresponding values from column "B" in a bar graph
    plt.bar(filtered_df["Language"], filtered_df["Mean WER"], color="skyblue")
    plt.title(f"Selected Languages Mean WER")
    plt.xlabel("Language")
    plt.ylabel("Mean WER")
    plt.show()


    return

def infer_test(audio_files):
    print('here')
    # Load pre-trained model and processor
    model_name = "facebook/wav2vec2-base-960h"  # Change to your preferred model
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    #model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained("./checkpoint-2500")
    model2 = Wav2Vec2ForCTC.from_pretrained(model_name)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    processor_checkpoint = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    language_listOfWER = {} # language --> list of WER gotten
    
    for (ind, audio_file) in enumerate(audio_files):
        if ind % 50 == 0:
            print('file num:', ind)
        print(audio_file)
        audio_file = "./speech_accent_archive/recordings/" + audio_file
        speech = preprocess_audio(audio_file)

        # Process the audio for input
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
            logits2 = model2(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids2 = torch.argmax(logits2, dim=-1)
        transcription = (processor_checkpoint.batch_decode(predicted_ids)[0]).upper()
        transcription2 = processor.batch_decode(predicted_ids2)[0]
        print("Transcription:", transcription)
        print('wer transcription1:', get_wer(transcription))
        print("Transcription2:", transcription2)
        #if (transcription == transcription2):
        #    print('theyre the same...')
        #else:
        #    print('theyre not the same')

def main():

    # get_data()
    #infer()

    files = list_mp3_files('./speech_accent_archive/recordings')
    #print(files)
    #infer_test(files)
    #infer_checkpoint_csv(files)

    """ language_count, filtered_language_count = gather_class_stats(files)
    print(language_count)

    key_count = 0
    value_count = 0
    for key,value in language_count.items():
        key_count += 1
        value_count += language_count[key]

    print(key_count, value_count) """

    """ files = list_mp3_files('./speech_accent_archive/recordings')
    print(files, len(files))

    language_count, filtered_language_count = gather_class_stats(files)

    # bar_graph_class_count(filtered_language_count)

    print(language_count)

    #infer(files)
    """
    """ english_items = [item for item in files if item.startswith("english")]
    spanish_items = [item for item in files if item.startswith("spanish")]
    arabic_items = [item for item in files if item.startswith("arabic")]
    filtered_files = [item for item in files if (not item.startswith("english")) and (not item.startswith("spanish")) and (not item.startswith("arabic"))]

    random.seed(42)
    print(filtered_files, len(filtered_files))

    english_items = random.sample(english_items, 65)
    spanish_items = random.sample(spanish_items, 65)
    arabic_items = random.sample(arabic_items, 65)

    new_list = filtered_files + english_items + spanish_items + arabic_items
    new_list = sorted(new_list, key=str.lower)

    print(new_list, len(new_list)) 

    random_selection = random.sample(new_list, 250)

    infer_checkpoint_csv(random_selection) """


    #language_count, filtered_language_count = gather_class_stats(new_list)
    #print(language_count)

    #infer(new_list)

    #df = pd.read_csv('Language_WER.csv')
    #sort(df)
    #df = pd.read_csv('Language_WER_Checkpoint.csv')
    #sort(df)

    #df = pd.read_csv('Language_WER_Checkpoint_Sorted.csv')
    #display_WER(df)

    infer_test(['spanish1.mp3'])

    return

if __name__ == "__main__":
    main()