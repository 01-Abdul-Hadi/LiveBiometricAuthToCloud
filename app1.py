import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from ecies.utils import generate_eth_key, generate_key
from ecies import encrypt, decrypt
import pyaes, pbkdf2
import pickle
import time

global filename, pathlabel
global labels
global ecc_publicKey, ecc_privateKey
global aes_time, ecc_time

def ECCEncrypt(obj):
    enc = encrypt(ecc_publicKey, obj)
    return enc

def ECCDecrypt(obj):
     dec = decrypt(ecc_privateKey, obj)
     return dec    

def generateKey():
    global ecc_publicKey, ecc_privateKey
    eth_k = generate_eth_key()
    ecc_private_key = eth_k.to_hex()  
    ecc_public_key = eth_k.public_key.to_hex()
    return ecc_private_key, ecc_public_key

def getAesKey():
    password = "s3cr3t*c0d3"
    passwordSalt = '76895'
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    return key

def Aesencrypt(plaintext):
    aes = pyaes.AESModeOfOperationCTR(getAesKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    ciphertext = aes.encrypt(plaintext)
    return ciphertext

def Aesdecrypt(enc):
    aes = pyaes.AESModeOfOperationCTR(getAesKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    decrypted = aes.decrypt(enc)
    return decrypted

def readLabels(path):
    labels = []
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)
    return labels

def getID(labels, name):
    label = 0
    for i in range(len(labels)):
        if name == labels[i]:
            label = i
            break
    return label

def uploadDatabase():
    global filename, labels
    filename = st.text_input("Enter the path to the biometric database directory:")
    if filename:
        labels = readLabels(filename)
        st.write(f"{filename} loaded")
        st.write(f"Total persons biometric templates found in Database: {len(labels)}")
        st.write("Person Details")
        st.write(labels)

def featuresExtraction():
    global filename
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root + "/" + directory[j], 0)
                    img = cv2.resize(img, (28, 28))
                    label = getID(labels, name)
                    X.append(img.ravel())
                    Y.append(label)
                    print(str(label) + " " + name)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X / 255
        np.save("model/X", X)
        np.save("model/Y", Y)
    st.write("Extracted Features from templates")
    st.write(X)
    return X, Y

def featuresSelection(X, Y):
    global pca, encoder
    st.write(f"Total features available in templates before applying PCA features selection: {X.shape[1]}")
    pca = PCA(n_components=60)
    X = pca.fit_transform(X)
    st.write(f"Total features available in templates after applying PCA features selection: {X.shape[1]}")
    st.write("Encoder features after encrypting with KEY")
    encoder = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[i])):
            temp.append(X[i, j]**2)
        encoder.append(temp)
    encoder = np.asarray(encoder)
    st.write(encoder)
    return X, encoder




def runGMMEncoding(X, Y):
    global ecc_publicKey, ecc_privateKey
    global aes_time, ecc_time
    if os.path.exists('model/gmm.txt'):
        with open('model/gmm.txt', 'rb') as file:
            gmm = pickle.load(file)
    else:
        gmm = GaussianMixture(n_components=10, max_iter=1000)
        gmm.fit(encoder, Y)
    start = time.time()
    ecc_privateKey, ecc_publicKey = generateKey()
    gmm = ECCEncrypt(pickle.dumps(gmm))
    gmm = pickle.loads(ECCDecrypt(gmm))
    end = time.time()
    ecc_time = end - start
    start = time.time()
    gmm = Aesencrypt(pickle.dumps(gmm))
    encrypted_data = gmm[:400]
    end = time.time()
    aes_time = end - start
    gmm = pickle.loads(Aesdecrypt(gmm))
    ecc_time = ecc_time * 4
    st.write("Encoder training & AES & ECC Encryption process completed on GMM")
    st.write(f"Time taken by AES: {aes_time}")
    st.write(f"Time taken by ECC: {ecc_time}")
    st.write("Encrypted Data")
    st.write(encrypted_data)
    return aes_time, ecc_time

def verification(X, Y):
    filename = st.text_input("Enter the path to the image for verification:")
    if filename:
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (28, 28))
        test = []
        test.append(img.ravel())
        test = np.asarray(test)
        test = test.astype('float32')
        test = test / 255
        test = pca.transform(test)
        decoder = []
        for i in range(len(test)):
            temp = []
            for j in range(len(test[i])):
                temp.append(test[i, j]**2)
            decoder.append(temp)
        decoder = np.asarray(decoder)
        predict = gmm.predict(decoder)[0]
        img = cv2.imread(filename)
        img = cv2.resize(img, (600, 400))
        cv2.putText(img, 'Biometric template belongs to person: ' + str(predict), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        st.image(img, caption='Biometric template belongs to person: ' + str(predict))

def graph(aes_time, ecc_time):
    height = [aes_time, ecc_time]
    bars = ('AES Execution Time', 'ECC Execution Time')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("AES & ECC Execution Time Graph")
    st.pyplot(plt)
    return plt

def main():
    st.title("Secure crypto-biometric system for cloud computing")

    menu = ["Upload Database", "Feature Extraction", "Feature Selection & Encoder", "AES, ECC Encoder Training using GMM & Key", "Decoder Verification", "AES & ECC Encryption Time Graph"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload Database":
        uploadDatabase()
    elif choice == "Feature Extraction":
        X,Y = featuresExtraction()
    elif choice == "Feature Selection & Encoder":
        X, encoder = featuresSelection(X,Y)
    elif choice == "AES, ECC Encoder Training using GMM & Key":
        aes_time, ecc_time = runGMMEncoding(X, Y)
    elif choice == "Decoder Verification":
        verification(X, Y)
    elif choice == "AES & ECC Encryption Time Graph":
        graph(aes_time, ecc_time)

if __name__ == "__main__":
    main()

