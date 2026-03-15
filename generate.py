import json
import random
import string

TAGS_2_ID = {
  "0": 0,
  "B_PROVINCE": 1,
  "I_PROVINCE": 2,
  "B_DISTRICT": 4,
  "I_DISTRICT": 5,
  "B_COMMUNE": 6,
  "I_COMMUNE": 7,
  "B_VILLAGE": 8,
  "I_VILLAGE": 9,
  "B_HOUSE": 10,
  "I_HOUSE": 11,
  "B_ROAD": 12,
  "I_ROAD": 13,
}

KHMER_CHARS = "".join(chr(i) for i in range(0x1780, 0x17EA))
KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
NOISE_CHARS = (
  KHMER_CHARS
  + string.ascii_letters
  + string.digits
  + "!@#$%^&*()_+~`-=[]{}|;':\",./<>?"
)

# Load Address Data
try:
  with open("data.json", encoding="utf-8") as fp:
    DATA = json.load(fp)
except FileNotFoundError:
  DATA = []

# Load Khmer Dictionary Words
try:
  with open("khmerdict.txt", "r", encoding="utf-8") as f:
    # Strip newlines and ignore empty lines
    KHMER_DICT_WORDS = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
  KHMER_DICT_WORDS = []


def encode_tag_value(value: str, tag: str):
  if not value:
    return []
  # If the tag is "0" (for our random dictionary words), label every character as "0"
  if tag == "0":
    return ["0"] * len(value)
  return ["B_" + tag] + (["I_" + tag] * (len(value) - 1))


def generate_house():
  prefixes = ["ផ្ទះលេខ", "ផ្ទះលេខ ", "ផ្ទះ", "ផ្ទះ ", "លេខ", "លេខ ", "No.", "No. ", ""]
  prefix = random.choice(prefixes)

  num_length = random.randint(1, 4)
  if random.random() < 0.5:
    number = "".join(random.choice(KHMER_DIGITS) for _ in range(num_length))
  else:
    number = "".join(random.choice(string.digits) for _ in range(num_length))

  if random.random() < 0.15:
    number += random.choice("ABCកខគឃង")

  return prefix + number


def generate_road():
  prefixes = ["ផ្លូវលេខ", "ផ្លូវលេខ ", "ផ្លូវ", "ផ្លូវ ", "St.", "St. ", "Street ", ""]
  prefix = random.choice(prefixes)

  if random.random() < 0.1:
    nat_prefix = random.choice(["ផ្លូវជាតិលេខ", "ផ្លូវជាតិលេខ "])
    nat_num = random.choice(KHMER_DIGITS[1:8] + string.digits[1:8])
    return nat_prefix + nat_num

  num_length = random.randint(1, 3)
  if random.random() < 0.5:
    number = "".join(random.choice(KHMER_DIGITS) for _ in range(num_length))
  else:
    number = "".join(random.choice(string.digits) for _ in range(num_length))

  return prefix + number


def generate_address(
  remove_space_prob=0.3,
  noise_prob=0.05,
  drop_char_prob=0.05,
  drop_prefix_prob=0.2,
  drop_component_prob=0.1,
  include_house_prob=0.8,
  include_road_prob=0.8,
  insert_word_prob=0.3,  # Probability to insert a random dictionary word
):
  for province in DATA:
    province_name = province["name"]["km"]
    for district in province["districts"]["values"]:
      district_name = district["name"]["km"]
      for commune in district["communes"]["values"]:
        commune_name = commune["name"]["km"]
        for village in commune["villages"]["values"]:
          village_name = village["name"]["km"]

          if not village_name.startswith("ភូមិ"):
            village_name = "ភូមិ" + village_name

          if random.random() < drop_prefix_prob:
            province_name = province_name.replace("ខេត្ត", "")
            province_name = province_name.replace("រាជធានី", "")

          if random.random() < drop_prefix_prob:
            district_name = district_name.replace("ស្រុក", "")
            district_name = district_name.replace("ខណ្ឌ", "")

          if random.random() < drop_prefix_prob:
            commune_name = commune_name.replace("ឃុំ", "")
            commune_name = commune_name.replace("សង្កាត់", "")

          if random.random() < drop_prefix_prob:
            village_name = village_name.replace("ភូមិ", "")

          components = [
            (province_name, "PROVINCE"),
            (district_name, "DISTRICT"),
            (commune_name, "COMMUNE"),
            (village_name, "VILLAGE"),
          ]

          if random.random() < include_road_prob:
            components.append((generate_road(), "ROAD"))
          if random.random() < include_house_prob:
            components.append((generate_house(), "HOUSE"))

          if random.random() < drop_component_prob:
            drop_idx = random.randint(0, len(components) - 1)
            del components[drop_idx]

          if random.random() > 0.5:
            components.reverse()

          # --- NEW: Insert random dictionary word ---
          if KHMER_DICT_WORDS and random.random() < insert_word_prob:
            random_word = random.choice(KHMER_DICT_WORDS)
            # Pick a random position to insert the word
            insert_idx = random.randint(0, len(components))
            # Tag as "0" so encode_tag_value knows to label it as non-entity
            components.insert(insert_idx, (random_word, "0"))

          text_chars = []
          label_tags = []

          for i, (name, tag_type) in enumerate(components):
            entity_tags = encode_tag_value(name, tag_type)

            for char, tag in zip(name, entity_tags):
              if random.random() < drop_char_prob:
                continue

              if random.random() < noise_prob:
                text_chars.append(random.choice(NOISE_CHARS))
                label_tags.append(tag)
              else:
                text_chars.append(char)
                label_tags.append(tag)

            if i < len(components) - 1:
              if random.random() >= remove_space_prob:
                text_chars.append(" ")
                label_tags.append("0")

          labels = [TAGS_2_ID[tag] for tag in label_tags]
          text = "".join(text_chars)

          yield text, labels


if __name__ == "__main__":
  address_generator = generate_address(
    remove_space_prob=0.0,
    noise_prob=0.0,
    drop_char_prob=0.0,
    drop_prefix_prob=0.0,
    drop_component_prob=0.0,
    insert_word_prob=1.0,  # Set to 1.0 here just to guarantee you see it in the test output
  )

  for _ in range(5):
    try:
      text, labels = next(address_generator)
      print(f"Text:   {text}")
      print(f"Labels: {labels}\n")
      print("-" * 50)
    except StopIteration:
      break
