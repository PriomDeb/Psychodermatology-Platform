import pandas as pd
import joblib
import pickle
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import base64
import os



def encrypt(password:str, object:any, name:str):
    password = password.encode() 
    
    # Derive a key from the password using PBKDF2
    salt = os.urandom(16)  # Salt for PBKDF2 (should be saved for decryption)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
        )
    
    key = base64.urlsafe_b64encode(kdf.derive(password))  # Derived key
    cipher_suite = Fernet(key)
    
    # Save the salt for decryption (necessary to derive the same key later)
    with open("salt_file.salt", "wb") as salt_file:
        salt_file.write(salt)
    
    # Convert DataFrame to binary using pickle or joblib
    df_binary = pickle.dumps(object)
    
    # Encrypt the binary data
    encrypted_data = cipher_suite.encrypt(df_binary)
    
    # Save the encrypted data to a joblib file
    with open(f"{name}", 'wb') as file:
        joblib.dump(encrypted_data, file)


def decrypt(password:str, object:any):
    password = password.encode()  # Same password used for encryption
    
    # Load the salt used during encryption
    with open("salt_file.salt", "rb") as salt_file:
        salt = salt_file.read()
    
    # Derive the same key from the password and salt
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
        )
    
    key = base64.urlsafe_b64encode(kdf.derive(password))  # Derived key
    cipher_suite = Fernet(key)
    
    # Load the encrypted data from the joblib file
    with open(f"{object}", 'rb') as file:
        encrypted_data = joblib.load(file)
    
    # Decrypt the binary data
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    
    # Convert the binary back to DataFrame
    df = pickle.loads(decrypted_data)
    
    return df



if __name__ == "__main__":
    file_path = "/Users/priom/Desktop/Psychodermatology"
    df = pd.read_excel(f"{file_path}/PsyDerm_new_final.xlsx")
    name="encrypted_df.joblib"
    
    from password import PASSWORD
    
    # encrypt(password=PASSWORD, object=df, name=name)
    # print(decrypt(password=PASSWORD, object="encrypted_df.joblib"))
    
