import camelot
import pdfplumber


def parse_zoning_def_dict(pdf_path):
    all_tables = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract raw text as lines
            lines = page.extract_text().splitlines()
            # Extract tables
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                # Find the position of the table in the raw text
                table_start_line = find_table_start(lines, table)
                # Extract the line before the table, if available
                label_line = lines[table_start_line - 2] if table_start_line > 0  else None
                table = [row for row in table if 'Abbreviation' not in row]
                if 'APPENDIX' in label_line:
                    label_line = re.sub('APPENDIX.*: ', '', label_line)
                    label_line = re.sub(' +', '_', label_line.lower())
                    prev_label_line = label_line
                elif 'ZONING TAX LOT DATA DICTIONARY' in label_line:
                    label_line = None
                elif 'APPENDIX' not in label_line:
                    table = [row for row in table if 'Abbreviation' not in row]
                if label_line != None:
                    all_tables[label_line] = table
                else:
                    all_tables[prev_label_line] = all_tables[prev_label_line] + table
    return all_tables

def find_table_start(lines, table):
    """
    Identify the start of the table in the text by matching table rows
    """
    for i, line in enumerate(lines):
        # Convert the table's first row into a string and search for it in the text
        table_row = " ".join(str(cell) for cell in table[1] if cell)  # Skip empty cells
        if table_row in line:
            return i
    return -1


####################################################################################################
##
##  Functions to extract zoning definitions from a particularly troublesome PDF, I doubt most of these will be reusable.
####################################################################################################

import pdfplumber
import camelot
import pandas as pd
import re
from collections import defaultdict
import traceback


def handle_sub_rows_cols(df):
    if any([cell == "" for cell in df.iloc[:, 0]]):
        for idx, i in enumerate(df.iloc[:, 0]):
            if i == "" and idx > 0:
                try:
                    df.iloc[idx, 0] = f"{df.iloc[:,0][idx - 1]}, {df.iloc[:,1][idx]}"
                    df.iloc[idx - 1, 0] = (
                        f"{df.iloc[:,0][idx - 1]}, {df.iloc[:,1][idx - 1]}"
                    )
                except Exception as e:
                    print(f"Error type: {type(e).__name__}, Error details: {e}")
                    formatted_traceback = traceback.format_exception(
                        type(e), e, e.__traceback__
                    )
                    # Print the full traceback
                    print("".join(formatted_traceback))
                    pass
    return df


def substitute_chars(df):
    df.replace({"\uf033": "True"}, regex=True, inplace=True)
    df.replace({"\u0016": "True"}, regex=True, inplace=True)
    df.replace({"(cid:22)": "True"}, regex=False, inplace=True)
    df.replace({"\n": " "}, regex=True, inplace=True)
    return df


def fix_table(df, orientation):
    df = handle_sub_rows_cols(df)
    df_title = df.iloc[0, :][0]
    df.drop(index=df.index[0], inplace=True)
    df.reset_index()
    df.columns = df.iloc[0, :]
    df.drop(index=df.index[0], inplace=True)
    df.index = df.iloc[:, 0]
    df.drop(columns=df.columns[0:1], inplace=True)
    df.name = df_title
    df.replace({r"\n": " "}, regex=True, inplace=True)
    try:
        df.drop(columns=[""], axis=1, inplace=True)
    except KeyError:
        pass
    df = substitute_chars(df)
    if orientation == "horizontal":
        return df
    else:
        return df.transpose()


def keep_lines_outside_table(page, lines):
    # Sort lines by their vertical position (top to bottom)
    sorted_lines = sorted(lines.items(), key=lambda x: x[0])
    # Get table bounding boxes
    table_bboxes = [table.bbox for table in page.find_tables()]
    table_orientations = []
    for bbox in table_bboxes:
        width_height_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        if width_height_ratio > 0.9:
            table_orientations.append("horizontal")
        else:
            table_orientations.append("vertical")
    # Process each line and exclude characters inside table bounding boxes
    filtered_text = []
    for _, line_chars in sorted_lines:
        line_text = ""
        for char in line_chars:
            char_x0, char_y0, char_x1, char_y1 = (
                char["x0"],
                char["top"],
                char["x1"],
                char["bottom"],
            )
            # Check if the character is inside any table bounding box
            in_table = any(
                char_x0 >= bbox[0]
                and char_x1 <= bbox[2]
                and char_y0 >= bbox[1]
                and char_y1 <= bbox[3]
                for bbox in table_bboxes
            )
            if not in_table:
                line_text += char["text"]
        # Add the filtered line to the results if it's not empty
        if line_text.strip():
            filtered_text.append(line_text)
    # Join lines with newline characters
    page_text = "\n".join(filtered_text)
    return table_orientations, page_text


# This is a pretty awkward and fragile set of functions, but I could not think of a way to extract the data from the pdf without this heavy customization.
# It deals with the case where there is a page with two tables, each table having its own set of footnotes, and those footnotes being arranged in two columns.
# The footnotes on the other pages are in just one column and could be extracted without resorting to looking for specific patterns.

def replace_footnote_num_with_footnote_text(df, table_footnotes, key, num):
    df.replace(
        {
            rf"<s>\s?{num}</s>": f" ({table_footnotes[key][num]})",
            "\u2013": "-",
        },
        regex=True,
        inplace=True,
    )
    df.replace(r"</?s>", "", regex=True, inplace=True)
    df.index = df.index.str.replace(
        rf"<s>\s?{num}</s>", f" ({table_footnotes[key][num]})", regex=True
    )
    df.columns = df.columns.str.replace(
        rf"<s>\s?{num}</s>", f" ({table_footnotes[key][num]})", regex=True
    )
    df.loc[
        "notes",
        df.columns[df.columns.str.contains(rf"<s>\s?{num}</s>", na=False)],
    ] = table_footnotes[key][num]
    df.index = df.index.str.replace(rf"<s>\s?{num}</s>", "", regex=True)
    df.columns = df.columns.str.replace(rf"<s>\s?{num}</s>", "", regex=True)
    df.index = df.index.str.replace(r" Inﬁ ll", "", regex=True)
    df.index = df.index.str.replace(r"</?s>", "", regex=True)
    df.index = df.index.str.replace(r"\n", " ", regex=True)
    df.columns = df.columns.str.replace("\n", " ", regex=True)
    return df

def parse_footnote_layout_1(key, page_content, orientation):
    table_footnotes = {key: {"footnotes": {}}}
    lines = page_content[key][1].split("\n")
    lines = [line.strip() for line in lines if line != ""]
    notes = re.split(r"\s\s+", lines[1])
    footnotes = {lines[0][0]: notes[0], lines[0][1]: notes[1]}
    table_footnotes[key]["footnotes"] = footnotes
    df = page_content[key][0].df
    df = fix_table(df, orientation)
    table_footnotes[key]["df"] = df
    return table_footnotes

def parse_footnote_layout_2(key, page_content, orientation):
    table_footnotes = {key : {'footnotes' : {}} }
    print(
        f"Assuming that {re.search(r'\n\d+\n\d+\n', page_content[key][1])} represents two columns of footnotes"
    )
    lines = page_content[key][1].split("\n")
    lines = [line.strip() for line in lines if line != ""]
    # Six or more spaces seems to identify the start of a line that is a continuation of a previous footnote
    for idx2, line in enumerate(lines):
        # 6 seems to be a reasonable heuristic for identifying lines with text content
        if line.endswith("     ") and len(lines[idx2 + 1]) > 6:
            lines[idx2] = lines[idx2].strip() + " " + lines[idx2 + 1]
            del lines[i + 1]
    for idx3, line in enumerate(lines):
        line = line.strip()
        try:
            if line.isdigit() and lines[idx3 + 1].isdigit():
                if key not in table_footnotes.keys():
                    table_footnotes[key] = {}
                num1 = line
                num2 = lines[idx3 + 1]
                table_footnotes[key]["footnotes"][num1] = lines[idx3 + 2].strip()
                table_footnotes[key]["footnotes"][num2] = lines[idx3 + 3].strip()
            else:
                if (
                    len(line) == 1
                    and line.isdigit()
                    and not lines[idx3 + 1].isdigit()
                    and len(lines[idx3 + 1]) > 6
                ):
                    # print("probably an addition to an existing footnote", lines[idx3 + 1])
                    if line not in table_footnotes[key].keys():
                        table_footnotes[key][line] = lines[idx3 + 1].strip()
                else:
                    continue
        except IndexError:
            print("End of the line at {idx3}?")
    df = page_content[key][0].df
    df = fix_table(df, orientation)
    df.loc["notes"] = ""
    for num in table_footnotes[key].keys():
        df = replace_footnote_num_with_footnote_text(df, table_footnotes, key, num)
    table_footnotes[key]["df"] = df
    return table_footnotes

def handle_two_col_footnotes(page_content, key, orientation):
    # table_footnotes = {key: {"footnotes": {}}}
    if re.match(r"^\n\d\d\n", page_content[key][1]):
        # table_footnotes = {key : {'footnotes' : {}} }
        print(
            f"Assuming that {re.search(r'\d\d', page_content[key][1])} represents two different footnotes separated by whitespaces. Also assuming only two footnotes"
        )
        table_footnotes = parse_footnote_layout_1(key, page_content, orientation)
    elif re.match(r"^\n\d+\n\d+\n", page_content[key][1]):
        table_footnotes = parse_footnote_layout_2(key, page_content, orientation)
    else:
        raise "Some sort of problem with handling two-column footnotes!"
    return table_footnotes


# This works for the pages that have one table and their footnotes in one column.
def parse_standard_page(footnote_text, key):
    footnotes = {}
    notes = footnote_text.strip()
    page_footnotes = {
        key: {"footnotes": {}},
    }
    for match in re.finditer(r"(\d+)\s+(.*)", footnote_text):
        number, content = match.groups()
        if number not in footnotes:
            footnotes[number] = content.strip()
        else:
            footnotes[number] += f" {content.strip()}"
    lines = notes.split("\n")
    number = None
    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line[0] == "©":
            continue
        # Check if this line starts a new footnote
        if len(stripped_line) == 1 and stripped_line[0].isdigit():
            number = stripped_line[0]
            page_footnotes[key]["footnotes"][number] = ""
        elif len(stripped_line) > 1:
            if number is not None:
                if page_footnotes[key]["footnotes"][number]:
                    page_footnotes[key]["footnotes"][number] = (
                        page_footnotes[key]["footnotes"][number] + stripped_line
                    )
                else:
                    page_footnotes[key]["footnotes"][number] = stripped_line
    return page_footnotes


def parse_multitable_page(page_text, lines, tables, table_orientations):
    results = {}
    split_by_table = re.split(r"(ZONING DATA TABLE \d+)", page_text)
    split_by_table = [text for text in split_by_table if text != ""]
    page_content = {
        split_by_table[0]: [tables[0], split_by_table[1]],
        split_by_table[2]: [tables[1], split_by_table[3]],
    }
    for idx, key in enumerate(page_content.keys()):
        table_and_footnotes = handle_two_col_footnotes(
            page_content, key, table_orientations[idx]
        )
        results |= table_and_footnotes
    return results

def parse_single_table_page(page_text, lines, table, table_orientation):
    # Stuff before the table name isn't needed
    split_content = re.split(r"(ZONING DATA TABLE \d+)", page_text.strip())
    split_content = [text for text in split_content if text != ""]
    if split_content[0].startswith("Zoning Data Tables"):
        del split_content[0]
    table_name = split_content[0].strip()
    page_content = parse_standard_page(split_content[1], table_name)
    df = table.df
    df = fix_table(df, table_orientation)
    df.loc["notes"] = ""
    for num in page_content[table_name]["footnotes"].keys():
        df.replace(
            {
                rf"<s>\s?{num}</s>": f' ({page_content[table_name]['footnotes'][num]})',
                "\u2013": "False",
            },
            regex=True,
            inplace=True,
        )
        df.index = df.index.str.replace(
            rf"<s>\s?{num}</s>",
            f' ({page_content[table_name]['footnotes'][num]})',
            regex=True,
        )
        df.loc[
            "notes",
            df.columns[
                df.columns.str.contains(rf"<s>\s?{num}</s>", na=False)
            ],
        ] = page_content[table_name]["footnotes"][num]
        df.index = df.index.str.replace(rf"<s>\s?{num}</s>", "", regex=True)
        df.columns = df.columns.str.replace(
            rf"<s>\s?{num}</s>", "", regex=True
        )
        df.index = df.index.str.replace("\n", " ", regex=True)
        df.columns = df.columns.str.replace("\n", " ", regex=True)
    page_content[table_name]["df"] = df
    # results |= page_content
    return page_content


def parse_zoning_details(pdf_path):
    """
    Extracts tables using Camelot and footnotes using pdfplumber, and matches footnotes to their corresponding tables.
    """
    results = {}

    # Extract footnotes using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = camelot.read_pdf(
                pdf_path,
                flavor="lattice",
                pages=str(page_num + 1),
                flag_size=True,
                line_scale=30,
            )
            # Extract all characters on the page
            chars = page.chars
            # Group characters by their approximate line position
            lines = defaultdict(list)
            for char in chars:
                line_key = round(
                    char["top"], 1
                )  # Use 'top' position to group characters into lines
                lines[line_key].append(char)
            table_orientations, page_text = keep_lines_outside_table(page, lines)
            # Note that this block assumes that a page with two tables also has the footnotes in two columns.
            # While that happens to be true for this pdf, any attempt to make this more reusable must address this.
            if len(re.findall(r"(ZONING DATA TABLE \d+)", page_text)) > 1:
                page_footnotes = parse_multitable_page(page_text, lines, tables, table_orientations)
                results |= page_footnotes
            elif len(re.findall(r"(ZONING DATA TABLE \d+)", page_text)) == 1:
                # Stuff before the table name isn't needed
                page_footnotes = parse_single_table_page(page_text, lines, tables[0], table_orientations[0])
                results |= page_footnotes
    return results


####################################################################################################
####################################################################################################
####################################################################################################
