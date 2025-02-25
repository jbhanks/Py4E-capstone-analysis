    print("Parsing table", table)
    rows = []
    k1,k2 = table[0][0].split(' ', 1)
    print('table[0][1][0][x0]', table[0][1][0]['x0'])
    col1_start = table[0][1][0]['x0']
    col2_start = table[0][1][1]['x0']
    print(f'col1_start: {col1_start}, col2_start :{col2_start}')
    current_line = 0 # Keep track of the line within the row, for handling table rows where a cell has multiple lines of text
    for i in table[1:]:
        print(f"i is {i}")
        if abs(i[1][0]['x0'] - col2_start) < 1:
            print("Continuing cell:", row)
            row = row + i[1]
            print("Continued cell:", row)
            current_line += 1
            continue
        elif current_line == 0:
            row = i[0].split(' ', 1)
            print("First line is", row)
            current_line += 1
            continue
        print("Row is", row)
        print('k1:', k1, 'k2:', k2, 'row:', row)
        rows.append({k1.lower(): row[0], k2.lower(): row[1] })