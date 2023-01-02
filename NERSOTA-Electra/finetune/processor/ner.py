import os
import copy
import json
import logging
import itertools
import ast

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("jieun0115/nersota-electra-small-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("jieun0115/nersota-electra-small-discriminator")


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def ner_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_seq_length,
        task,
        pad_token_label_id=-100,
):
    label_lst = ner_processors[task](args).get_labels()
    label_map = {label: i for i, label in enumerate(label_lst)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example {} of {}".format(ex_index, len(examples)))

        # tokens = []
        label_ids = []

        tokens = example.words

        for label in example.labels:
            label_ids.append(label_map[label])
            # word_tokens = word
            # if not word_tokens:
            #     word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            # tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids)
        )
    return features


class NerProcessor(object):
    """Processor for the NER data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ['O', 'TR-B','TR-I','FD-B','FD-I','QT-B','QT-I','AM-B','AM-I','MT-B','MT-I','AF-B','AF-I','PS-B','PS-I','LC-B','LC-I','EV-B','EV-I','OG-B','OG-I','DT-B','DT-I','TI-B','TI-I','TM-B','TM-I','CV-B','CV-I','PT-B','PT-I']

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, dataset, set_type):
        import ast
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, data) in enumerate(dataset):
            words, labels = data.split('\t')

            labels = ast.literal_eval(labels)
            try:
                labels = ast.literal_eval(labels)
            except:
                pass

            tokens = tokenizer.tokenize(words)
            string = words.split() # 띄어쓰기 단위로 split
            
            # making label sequences w/spacing
            total = []
            for idx, s in enumerate(string):
                s = tokenizer.tokenize(s)
                total.append(s)
                if idx != len(string)-1:
                    total.append(' ')
            total = list(itertools.chain.from_iterable(total))

            prefix_sum_of_token_start_index = []
            sum = 0

            for i, token in enumerate(total):
                if token == ' ':
                    sum += len(token)
                else:
                    if i == 0:
                        prefix_sum_of_token_start_index.append(0)
                        sum += len(token)
                    elif '##' in token:
                        prefix_sum_of_token_start_index.append(sum)
                        sum += len(token) - 2
                    else:
                        prefix_sum_of_token_start_index.append(sum)
                        sum += len(token)
            
            list_of_ner_tag = []
            list_of_ner_text = []
            list_of_tuple_ner_start_end = []

            for element in labels:
                try:
                    list_of_ner_tag.append(element['labels'])
                    list_of_ner_text.append(element['text'])
                    list_of_tuple_ner_start_end.append((element['start'], element['end']))
                except:
                    pass

            list_of_ner_label = []
            entity_index = 0
            is_entity_still_B = True

            for tup in zip(tokens, prefix_sum_of_token_start_index):
                token, index = tup

                if entity_index < len(list_of_tuple_ner_start_end):
                    start, end = list_of_tuple_ner_start_end[entity_index]

                    if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                        is_entity_still_B = True
                        entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                        start, end = list_of_tuple_ner_start_end[entity_index]

                    if start <= index and index < end:  
                        entity_tag = list_of_ner_tag[entity_index]
                        if is_entity_still_B is True:
                            entity_tag = entity_tag + '-B'
                            list_of_ner_label.append(entity_tag)
                            is_entity_still_B = False
                        else:
                            entity_tag = entity_tag + '-I'
                            list_of_ner_label.append(entity_tag)
                    else:
                        is_entity_still_B = True
                        entity_tag = 'O'
                        list_of_ner_label.append(entity_tag)

                else:
                    entity_tag = 'O'
                    list_of_ner_label.append(entity_tag)

            words = tokens
            labels = list_of_ner_label
            guid = "%s-%s" % (set_type, i)

            assert len(words) == len(labels)

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir,
                                                        self.args.task,
                                                        file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir,
                                                                  self.args.task,
                                                                  file_to_read)), mode)


ner_processors = {
    "ner": NerProcessor
}

ner_tasks_num_labels = {
    "ner": 31
}


def ner_load_and_cache_examples(args, tokenizer, mode):
    processor = ner_processors[args.task](args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")

        pad_token_label_id = CrossEntropyLoss().ignore_index
        features = ner_convert_examples_to_features(
            args,
            examples,
            tokenizer,
            max_seq_length=args.max_seq_len,
            task=args.task,
            pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return dataset 