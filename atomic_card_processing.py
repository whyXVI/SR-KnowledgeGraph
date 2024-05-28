import os
import openai
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, NamedTuple, Set
import scipy
import time
import glob

from basic_utils import *


def save_card_df_to_json(card_df, save_file_name):
    card_df_as_json = card_df.to_json(orient="index")

    try:
        os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
        with open(save_file_name + '.json', 'w') as f:
            json.dump(card_df_as_json, f)
    except:
        with open(save_file_name + '.json', 'w') as f:
            json.dump(card_df_as_json, f)


def save_card_df_to_json_utf(card_df, save_file_name):
    card_df_as_json = card_df.to_json(orient="index", force_ascii=False)
    json_data = json.loads(card_df_as_json)

    try:
        os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
        with open(save_file_name + '.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    except: # when it accepts not a dirname
        with open(save_file_name + '.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)


def merge_and_save(csv_title, error_mask=None, start_ind=None):
    """
    This function takes in atomic jsons, merge them if it exists and is not cancelled
    and saves it to a sorted json, reserving original indexes
    """
    dfs = []
    num_pat = re.compile(fr"{csv_title}_card_(\d+)\.json")
    file_pat = f"./{csv_title}/{csv_title}_card_*.json"

    for filepath in glob.glob(file_pat):
        filename = os.path.basename(filepath)
        ind = num_pat.match(filename).group(1)

        if error_mask is not None and int(ind) in error_mask:
            print(f"Skipping ind_{ind}")
            continue
        if start_ind:
            if ind < start_ind:
                print('pass', ind)
                continue

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_dict({ind: data["0"]}, orient="index")
        dfs.append(df)

    card_pd = pd.concat(dfs)

    card_pd.index = card_pd.index.astype(int)
    card_pd.sort_index(ascending=True, inplace=True)
    card_pd.index = card_pd.index.astype(str)

    save_card_df_to_json_utf(card_pd, csv_title + '_cards_df_abstraction_groups')
    print('saved aggregated deck', csv_title + '_cards_df_abstraction_groups' + '.json\nAggregated', len(card_pd))
    return None


def read_card_df_from_json(save_file_name):
    with open(save_file_name + '.json', 'r', encoding='utf-8') as f:
        card_df_reloaded = pd.read_json(json.load(f), orient="index")
    return card_df_reloaded


def get_card_df_text_descriptions_from_front_and_back(flashcardExample_front,
                                                      flashcardExample_back,
                                                      verbose=False):
    flashcard_text_descriptions = []
    # global_tokens_used_for_card_reading = 0  # to keep track
    #
    # for card_ind in range(len(flashcardExamples_front)):
    #     if card_ind % 20 == 0:
    #         print('Card index: ', card_ind)
    #         print("global_tokens_used_for_card_reading: ", global_tokens_used_for_card_reading)

    total_used_tokens = 0

    flashcardQuestion = flashcardExample_front
    flashcardAnswer = flashcardExample_back
    ##########################

    flashcardPrompt = (
                "Translate the following text into English only, according to its meaning:\n" + flashcardQuestion)
    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=50)
    flashcardQuestion = response_text
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    flashcardPrompt = ("Translate the following text into English only, according to its meaning:\n" + flashcardAnswer)
    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=150)
    flashcardAnswer = response_text
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    ##########################

    flashcardIntroduction = "Here is a flashcard written by Professor Smith for the students in his class to study: \n"
    flashcardStatement = ("\nFlashcard Content:\nQuestion: { " + flashcardQuestion + " } \n" +
                          "Answer: { " + flashcardAnswer + " } \n")
    flashcardRephraseRequestKeyIdeas = (
                "\nProfessor Smith has made a list of the key facts and ideas a student must know " +
                "in order to fully understand this flashcard. " +
                "He has taken care to explain all acronyms and abbreviations, and and made no assumptions about the knowledge of the student. " +
                "To be even more helpful, he has formatted the list of ideas in a structured way to convey the hierarchy of ideas being tested, " +
                "such as into numbered lists, when applicable. Overall, he has tried to make the answer brief and concise, " +
                "while maintaining completeness and not introducing ambiguity.\n\n" +
                "Professor Smith's numbered list of key ideas necessary to understand the flashcard:\n")
    flashcardPrompt = (flashcardIntroduction +
                       flashcardStatement +
                       flashcardRephraseRequestKeyIdeas)

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=400)
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    flashcardKeyIdeas = "\nKey ideas necessary to understand the flashcard:\n" + response_text + "\n"
    flashcardKeyIdeas_list = response_text

    ##########################
    flashcardRephraseRequestSubjects = (
                "\nProfessor Smith has also compiled a numbered list of the minor subjects discussed within the flashcard. " +
                "These are the component objects which make up the major ideas of the flashcard. "
                "These are, for example, the nouns and objects discussed in both the question and answer. "
                "The names of subjects in the list are reported in extremely brief form (less than 3 words, preferably 1 word), "
                "since we want to compare them to other flashcards.\n\n" +
                "Professor Smith's numbered list of subjects discussed in the flashcard:\n")

    flashcardPrompt = (flashcardIntroduction +
                       flashcardStatement +
                       flashcardStatement +
                       flashcardKeyIdeas +
                       flashcardRephraseRequestSubjects)

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=400)
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    flashcardSubjects = "\nSubjects discussed in the flashcard:\n" + response_text + "\n"
    flashcardSubjects_list = response_text

    ##########################
    flashcardRephraseRequestSummary = (
            "\nProfessor Smith has also written a summary of the contents of the flashcard, to help his students understand the context of its information. " +
            "He categorizes the topic of the flashcard, and then the abstract category that topic is contained in, followed by more abstract categories " +
            "that these categories are contained in, in increasing order of abstraction.\n\n" +
            "His explanation uses the following format:\n" +
            "Specific topic: [text]  # names should be brief, less than 3 words, but preferably one word, and capitalized. Names should not be abbreviations.\n" +
            "General category: [[text], [text], ... ]\n" +
            "More general categories: [[text], [text], ... ]\n" +
            "Even more general categories: [[text], [text], ... ]\n" +
            "Most general category: [[text], [text], ... ]\n\n" +
            "Professor Smith's summary:\n")

    flashcardPrompt = (flashcardIntroduction +
                       flashcardStatement +
                       flashcardKeyIdeas +
                       flashcardSubjects +
                       flashcardRephraseRequestSummary)

    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=400)
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    flashcardExpandedExplanation = response_text

    #############################################################################################
    # Save text data for later parsing and conversion to dictionary
    combined_dict = {}
    combined_dict["Question"] = flashcardQuestion
    combined_dict["Answer"] = flashcardAnswer
    combined_dict["Key ideas"] = flashcardKeyIdeas_list
    combined_dict["flashcardSubjects_list"] = flashcardSubjects_list
    combined_dict["flashcardExpandedExplanation"] = flashcardExpandedExplanation
    # pprint(combined_dict)
    flashcard_text_descriptions.append(combined_dict)

    # print("Total used tokens:", total_used_tokens, " for card index ", card_ind)
    # global_tokens_used_for_card_reading += total_used_tokens

    card_df_text_description = pd.DataFrame(flashcard_text_descriptions)

    return card_df_text_description, total_used_tokens


def get_card_df_meta_data_from_text_description(card_df_text_description, card_ind,
                                                verbose=False):
    """
    Returns a pandas dataframe with the text descriptions converted into dictionaries
    which contain the separated out key words at various levels of abstraction.
    """

    flashcard_list_of_dicts = []
    # global_tokens_used_for_card_reading = 0  # to keep track

    # for card_ind in range(len(cards_df_text_descriptions)):
    #     if card_ind % 20 == 0:
    #         print('Card index: ', card_ind)
    #         print("global_tokens_used_for_card_reading: ", global_tokens_used_for_card_reading)

    total_used_tokens = 0

    ##########################

    flashcardSubjects_list = card_df_text_description["flashcardSubjects_list"].values[0]
    flashcardExpandedExplanation = card_df_text_description["flashcardExpandedExplanation"].values[0]

    ##########################

    # Now extract meta data to JSON files
    jsonConversionFailure = False
    flashcardPrompt = (
            "Reformat the following subject_list into a JSON array. Capitalize the first character of each word. " +
            'Set all plural words to singular form. For example "car mechanics" should become "Car Mechanic". '
            'Transfer plural to singular, and capitalization. ' +
            '"Flights" should become "Flight" (plural to singular), and "new World Machines" should become "New World '
            'Machine"\n\n' +
            "subject_list:\n" +
            flashcardSubjects_list +
            "\n\nResult:\n")
    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=400)
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    try:
        subject_list = json.loads(response_text)

        # check datatype for everything
        if not isinstance(subject_list, list):
            jsonConversionFailure = True
            print("   !!!!!!!! JSON conversion failed (not a real list of subjects)", " for card index ", card_ind)

    except:
        jsonConversionFailure = True
        print("   !!!!!!!! JSON conversion failed", " for card index ", card_ind)

    flashcardPrompt = (
            "Reformat the following information into a JSON dictionary containing lists of strings.\n\n" +
            "Information:\n{\n" +
            flashcardExpandedExplanation +
            "\n}\n\n" +
            "Use the following format:\n{\n" +
            '"Specific topic": [your text here, your text here, ...],  # Place items from "Specific topic" here, '
            'as a list of strings.\n' +
            '"General category": [your text here, your text here, ...],  # Place items from "General category" here, '
            'as a list of strings.\n' +
            '"More general categories": [],  # Place items from "More general categories" here" here, as a list of '
            'strings.\n' +
            '"Even more general categories": [],  # Place items from "Even more general categories" here, as a list '
            'of strings.\n' +
            '"Most general category": []  # Place items from "Most general category" here, as a list of strings.\n' +
            "\n}\n\n" +
            "For each string in the list, set all plural words to singular form, and capitalize the first character "
            "of each word. " +
            'For example, a specific topic of "car mechanics" should become "Car Mechanic" (plural to singular, '
            'and capitalization), ' +
            '"Flights" should become "Flight" (plural to singular), and "new World Machines" should become "New World '
            'Machine.\n\n' +
            "Result:\n")
    response_text, used_tokens = gen_response_text_with_backoff(flashcardPrompt, max_tokens=400)
    total_used_tokens += used_tokens  # print("Used tokens:",used_tokens)
    if verbose:
        print(flashcardPrompt + response_text)

    try:
        dict_of_abstractions = json.loads(response_text)

        # check datatype for everything
        if not isinstance(dict_of_abstractions, dict):
            jsonConversionFailure = True
            print("   !!!!!!!! JSON conversion failed (not a real dict of abstractions)", " for card index ",
                  card_ind)

        if isinstance(dict_of_abstractions, dict):
            for k, v in dict_of_abstractions.items():
                if not (isinstance(k, str) and isinstance(v, list)):
                    jsonConversionFailure = True
                    print(
                        "   !!!!!!!! JSON conversion failed (abstraction dict doesn't contain all str:list pairs)",
                        " for card index ", card_ind)
                for _hopefully_a_string in v:
                    if not isinstance(_hopefully_a_string, str):
                        jsonConversionFailure = True
                        print(
                            "   !!!!!!!! JSON conversion failed (abstraction dict doesn't contain a list of strings)",
                            " for card index ", card_ind)

    except:
        jsonConversionFailure
        print("   !!!!!!!! JSON conversion failed", " for card index ", card_ind)

    ##########################

    # Save data to a dictionary, then append to flashcard_list_of_dicts

    if not jsonConversionFailure:  # then save
        combined_dict = dict_of_abstractions.copy()
        combined_dict["Question"] = card_df_text_description["Question"].values[0]
        combined_dict["Answer"] = card_df_text_description["Answer"].values[0]
        combined_dict["Subjects"] = subject_list
        combined_dict["Key ideas"] = card_df_text_description["Key ideas"].values[0]
        # pprint(combined_dict)
        flashcard_list_of_dicts.append(combined_dict)

    # print("Total used tokens:", total_used_tokens, " for card index ", card_ind)

    card_df_meta_data = pd.DataFrame(flashcard_list_of_dicts)

    return card_df_meta_data, total_used_tokens


def get_and_save_cards_df(flashcardExamples_front,
                          flashcardExamples_back,
                          verbose=False):
    global_tokens_used_for_card_reading = 0  # to keep track

    for card_ind in range(len(flashcardExamples_front)):
        if card_ind % 20 == 0:
            print('Card index: ', card_ind)
            print("global_tokens_used_for_card_reading: ", global_tokens_used_for_card_reading)
        df1 = get_card_df_text_descriptions_from_front_and_back(flashcardExamples_front[card_ind],
                                                                flashcardExamples_back[card_ind],
                                                                verbose=verbose)
        get_card_df_meta_data_from_text_description(df1, card_ind, verbose=verbose)


def get_card_df_abstraction_groups_from_meta_data(cards_df):
    """
    Converts meta data with weird names for abstraction levels into a common format
    """

    new_cards_df = pd.DataFrame({})

    # Load in basic info
    new_cards_df["Question"] = cards_df["Question"].values
    new_cards_df["Answer"] = cards_df["Answer"].values
    new_cards_df["Key ideas"] = cards_df["Key ideas"].values

    # More complicated info
    new_cards_df["Abstraction groups"] = [{}]

    # Add cards
    abstraction_group_dict = new_cards_df["Abstraction groups"].values[0]

    abstraction_group_dict['-1'] = cards_df['Subjects'].values[0]
    abstraction_group_dict['0'] = cards_df['Specific topic'].values[0]
    abstraction_group_dict['1'] = cards_df['General category'].values[0]
    abstraction_group_dict['2'] = cards_df["More general categories"].values[0]
    abstraction_group_dict['3'] = cards_df['Even more general categories'].values[0]
    abstraction_group_dict['4'] = cards_df['Most general category'].values[0]

    return new_cards_df


def get_card_df_abstraction_groups_from_front_and_back_list(flashcardExample_front,
                                                            flashcardExample_back,
                                                            card_ind,
                                                            verbose=False):
    card_df_text_description, total_used_tokens1 = get_card_df_text_descriptions_from_front_and_back(
        flashcardExample_front, flashcardExample_back, verbose=verbose)
    card_df_meta_data, total_used_tokens2 = get_card_df_meta_data_from_text_description(card_df_text_description,
                                                                                        card_ind, verbose=verbose)
    new_card_df = get_card_df_abstraction_groups_from_meta_data(card_df_meta_data)

    print(total_used_tokens2 + total_used_tokens1, 'tokens in total used for card', card_ind)

    return new_card_df


def get_cards_df_abstraction_groups_from_front_and_back_csv(csv_title, verbose=False, start_ind=None):
    # read in raw front and back sets of flashcards
    cards_raw_front_and_back_df = pd.read_csv(csv_title + '.csv')
    flashcardList_front_text = cards_raw_front_and_back_df['front'].values
    flashcardList_back_text = cards_raw_front_and_back_df['back'].values
    for card_ind in range(len(flashcardList_front_text)):
        if card_ind < start_ind:
            continue
        card_df_abstraction_groups = get_card_df_abstraction_groups_from_front_and_back_list(
            flashcardList_front_text[card_ind],
            flashcardList_back_text[card_ind],
            card_ind,
            verbose=verbose)

        filename = f"./{csv_title}/{csv_title}_card_{str(card_ind)}"
        save_card_df_to_json_utf(card_df_abstraction_groups, filename)

    return None
