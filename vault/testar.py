import hvac

vault_url = 'http://localhost:1234'
unseal_key = 'jz1IJJa5jTVmldCCe2s3wB7kKZpWiJraIa9tb8YeWAw='
root_key = 'myroot'

client = hvac.Client(url=vault_url, token=root_key)

print(client.is_authenticated())

oi = client.read('/secret/data/oi')["data"]['data']
print(oi)
print(oi['a'])

# client.write('secret/snakes', type='pythons', lease='1h')
# print(client.read('secret/snakes'))
