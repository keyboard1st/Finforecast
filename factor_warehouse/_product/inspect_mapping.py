from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from get_data_from_DB import get_bond_stk_name_dict

if __name__ == '__main__':
    mapping = get_bond_stk_name_dict()
    print('bond_code -> stock_code 映射（前20项）：')
    for i, (bond, stock) in enumerate(mapping.items()):
        print(f'{i+1}: bond: {bond!r}, stock: {stock!r}')
        if i >= 19:
            break 