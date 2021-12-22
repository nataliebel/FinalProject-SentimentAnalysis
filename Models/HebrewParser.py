import requests

Token = "674cc7b4d6afe153ba825e82f3fce6e5"

def get_parsed_heb_text(text):
    # Escape double quotes in JSON.
    text = text.replace(r'"', r'\"')
    url = 'https://www.langndata.com/api/heb_parser?token=674cc7b4d6afe153ba825e82f3fce6e5'
    json = '{"data":"' + text + '"}'
    r = requests.post(url, data=json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})
    print(r.json()["lemmas"])
    return r.json()["lemmas"]