import binascii
from web3.auto import w3

key_filename = "/tmp/go-ethereum-keystore033256609/UTC--2019-07-23T13-33-05.609576239Z--8a62181ee6d3ed6cdc855e6eec2e93cc891738ca"
password = ''
with open(key_filename) as keyfile:
    encrypted_key = keyfile.read()
    private_key = w3.eth.account.decrypt(encrypted_key, password)
    print(binascii.b2a_hex(private_key))