import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "https://luggageadvisor.uk"
user = "Emily"
password = "VWgS KrsU mstl zSHO ZmUq e7ND"

r = requests.get(f"{BASE_URL}/wp-json/wp/v2/users/me", auth=HTTPBasicAuth(user, password))
print(r.status_code, r.text)

r = requests.post(
    f"{BASE_URL}/wp-json/wp/v2/posts",
    auth=HTTPBasicAuth(user, password),
    json={"title": "REST test post", "status": "draft"}
)
print(r.status_code, r.text)