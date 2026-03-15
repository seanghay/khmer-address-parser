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
NOISE_CHARS = (
  KHMER_CHARS
  + string.ascii_letters
  + string.digits
  + "!@#$%^&*()_+~`-=[]{}|;':\",./<>?"
)

with open("data.json") as fp:
  DATA = json.load(fp)


def encode_tag_value(value: str, tag: str):
  if not value:
    return []
  return ["B_" + tag] + (["I_" + tag] * (len(value) - 1))


def generate_address(
  remove_space_prob=0.3,
  noise_prob=0.05,
  drop_char_prob=0.05,
  drop_prefix_prob=0,
  drop_component_prob=0,
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

          # Group components for easier sequential processing

          if random.random() < drop_prefix_prob:
            province_name = province_name.replace("ខេត្ត", "")
            province_name = province_name.replace("រាជធានី", "")

          if random.random() < drop_prefix_prob:
            district_name = district_name.replace("ស្រុក", "")
            district_name = district_name.replace("សង្កាត់", "")

          if random.random() < drop_prefix_prob:
            commune_name = commune_name.replace("ឃុំ", "")
            commune_name = commune_name.replace("ខណ្ឌ", "")

          if random.random() < drop_prefix_prob:
            village_name = village_name.replace("ភូមិ", "")

          components = [
            (province_name, "PROVINCE"),
            (district_name, "DISTRICT"),
            (commune_name, "COMMUNE"),
            (village_name, "VILLAGE"),
          ]

          if random.random() < drop_component_prob:
            drop_idx = random.randint(0, len(components) - 1)
            del components[drop_idx]

          if random.random() > 0.5:
            components.reverse()

          text_chars = []
          label_tags = []

          for i, (name, tag_type) in enumerate(components):
            entity_tags = encode_tag_value(name, tag_type)

            # 1. Handle character-level dropping and noise
            for char, tag in zip(name, entity_tags):
              # Check if we should drop the character entirely
              if random.random() < drop_char_prob:
                continue  # Skip appending both char and tag

              # If not dropped, check if we should substitute with noise
              if random.random() < noise_prob:
                # Substitute with a noisy character but KEEP the original entity label.
                # This teaches the model to recognize the entity despite typos.
                text_chars.append(random.choice(NOISE_CHARS))
                label_tags.append(tag)
              else:
                text_chars.append(char)
                label_tags.append(tag)

            # 2. Handle spaces between components
            if i < len(components) - 1:
              # Only insert a space if we don't trigger the removal probability
              if random.random() >= remove_space_prob:
                text_chars.append(" ")
                label_tags.append("0")

          # Map string tags to their integer IDs
          labels = [TAGS_2_ID[tag] for tag in label_tags]
          text = "".join(text_chars)

          yield text, labels


if __name__ == "__main__":
  # Example: 30% chance to remove spaces, 10% chance for noisy characters, 5% chance to drop a character
  address_generator = generate_address(
    remove_space_prob=0.0,
    noise_prob=0.0,
    drop_char_prob=0.0,
    drop_prefix_prob=0.0,
    drop_component_prob=0.0,
  )

  # Print the first 5 examples to see the augmentation in action
  for _ in range(5):
    try:
      text, labels = next(address_generator)
      print(f"Text:   {text}")
      print(f"Labels: {labels}\n")
      print("-" * 50)
    except StopIteration:
      break
