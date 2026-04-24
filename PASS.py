#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:46:00 2024

@author: jamesz
"""


class SimpleCipher:
    def __init__(self):
        # Fixed encryption mapping
        self.mapping = {
            'a': 'q', 'b': 'w', 'c': 'e', 'd': 'r', 'e': 't',
            'f': 'y', 'g': 'u', 'h': 'i', 'i': 'o', 'j': 'p',
            'k': 'a', 'l': 's', 'm': 'd', 'n': 'f', 'o': 'g',
            'p': 'h', 'q': 'j', 'r': 'k', 's': 'l', 't': 'z',
            'u': 'x', 'v': 'c', 'w': 'v', 'x': 'b', 'y': 'n',
            'z': 'm', 'A': 'Q', 'B': 'W', 'C': 'E', 'D': 'R',
            'E': 'T', 'F': 'Y', 'G': 'U', 'H': 'I', 'I': 'O',
            'J': 'P', 'K': 'A', 'L': 'S', 'M': 'D', 'N': 'F',
            'O': 'G', 'P': 'H', 'Q': 'J', 'R': 'K', 'S': 'L',
            'T': 'Z', 'U': 'X', 'V': 'C', 'W': 'V', 'X': 'B',
            'Y': 'N', 'Z': 'M', '0': '5', '1': '4', '2': '3',
            '3': '2', '4': '1', '5': '0', '6': '9', '7': '8',
            '8': '7', '9': '6'
        }
        # Create reverse mapping
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def encrypt(self, plaintext):
        encrypted_text = ""
        for char in plaintext:
            encrypted_text += self.mapping.get(char, char)
        return encrypted_text

    def decrypt(self, ciphertext):
        decrypted_text = ""
        for char in ciphertext:
            decrypted_text += self.reverse_mapping.get(char, char)
        return decrypted_text
def main():
    cipher = SimpleCipher()
    while True:
        print("Select an option:")
        print("1. Encrypt")
        print("2. Decrypt")
        option = input("Enter your choice: ")

        if option == "1":
            plaintext = input("Enter the string to encrypt: ")
            encrypted_text = cipher.encrypt(plaintext)
            print("Encrypted:", encrypted_text)
            break
        elif option == "2":
            ciphertext = input("Enter the string to decrypt: ")
            decrypted_text = cipher.decrypt(ciphertext)
            print("Decrypted:", decrypted_text)
            break
        else:
            print("Invalid option. Please select either 1 or 2.")



if __name__ == "__main__":
    main()
