from google.oauth2 import service_account


def GetCredentials():
    service_account_key_path="/Users/kade.chen/go-kade-project/github/mcenter/etc/kade-poc.json"
    credentials = service_account.Credentials.from_service_account_file(
    service_account_key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
    return credentials

if __name__ == "__main__":
    # CreateImportDataset(bqclient,os.getenv("PROJECT_ID_Genai"), os.getenv("DATASET_ID"), os.getenv("TABLE_ID"))
   GetCredentials() 