import configparser
from ansible_vault import Vault
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from typing import Dict

class AnsibleVaultError(Exception):
    def __init__(self, comment):
        self.comment = comment
    
    def __str__(self):
        return self.comment
        

class AnsibleVault:
    secrets: Dict = None

    def __init__(self, vault_file: str):
        self.vault = Vault(os.getenv('ANSIBLE_VAULT_PWD'))
        self.vault_file = vault_file     

        with open(self.vault_file) as f:
            self.secrets = self.vault.load(f.read())

    def get_all_secrets(self):
        return self.secrets
    
    def get_secret(self, name: str):
        try:
            return self.secrets[name]
        except KeyError:
            raise AnsibleVaultError(f"Secret with name {name} does not exist in the vault")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    vault_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault'])
    ansible_vault = AnsibleVault(vault_file)

    print(ansible_vault.get_all_secrets())