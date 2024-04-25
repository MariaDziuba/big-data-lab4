import configparser
from ansible_vault import Vault
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)


class AnsibleVault:
    def __init__(self, pwd_file: str, vault_file: str):
        with open(pwd_file) as f:
            self.vault = Vault(f.read())
        self.vault_file = vault_file

    def get_secrets(self):
        with open(self.vault_file) as f:
            return self.vault.load(f.read())


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    vault_pwd_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault_pwd'])
    vault_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault'])
    ansible_vault = AnsibleVault(vault_pwd_file, vault_file)

    print(ansible_vault.get_secrets())