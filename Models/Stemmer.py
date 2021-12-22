import requests

def get_base_sentence_heb(sentences):
  request = {
    'token': 'gOsOnJ4XO4B3A6I',
    'readable': False,
    'paragraph': u'מה קורה'
  }
  result = requests.post('https://hebrew-nlp.co.il/service/Morphology/Analyze', json=request).json()

  base_sentences = []
  for sentence in result:
    new_sentence = ""
    for word in sentence:
      best_option = word[0]
      new_sentence += best_option['baseWord'] + ' '
    base_sentences.append(new_sentence)
  return base_sentences