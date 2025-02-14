import os
import json
import re

def process_dot_bracket_notation(question):
    """
    在点括号表示法中的每个字符后添加换行符
    """
    # 查找连续的点和括号序列
    pattern = r'[.()]{10,}'  # 匹配10个或更多连续的点和括号
    
    def replace_dot_bracket(match):
        sequence = match.group(0)
        return '\n'.join(list(sequence)) + '\n'
    
    return re.sub(pattern, replace_dot_bracket, question)

def process_sequence_in_question(question):
    """
    处理问题中的序列和点括号表示法
    """
    def replace_sequence(match):
        sequence = match.group(0)
        # 只处理长度大于7的连续大写字母序列
        if len(sequence) > 7:
            return '\n'.join(list(sequence)) + '\n'
        return sequence

    # 首先处理生物序列
    # 分割问题中可能的多个序列（以逗号分隔）
    parts = re.split(r'([A-Z]{6,})', question)
    processed_parts = []
    
    for part in parts:
        if re.match(r'^[A-Z]{7,}$', part):  # 如果是长度大于7的连续大写字母序列
            processed_parts.append('\n'.join(list(part)) + '\n')
        else:
            processed_parts.append(part)
    
    # 处理完生物序列后的结果
    intermediate_result = ''.join(processed_parts)
    
    # 然后处理点括号表示法
    final_result = process_dot_bracket_notation(intermediate_result)
    
    return final_result

def process_json_files(base_directory):
    """
    处理指定的生物相关文件夹中的JSON文件
    """
    # 指定要处理的文件夹
    target_folders = ['Drug-benchmark', 'Protein-benchmark', 'RBP-benchmark', 'RNA-benchmark']
    modified_count = 0
    total_files = 0
    
    print(f"Starting to process files in specified folders")
    
    # 遍历目标文件夹
    for folder in target_folders:
        folder_path = os.path.join(base_directory, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder}")
            continue
            
        print(f"\nProcessing folder: {folder}")
        
        # 遍历文件夹中的所有JSON文件
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                file_path = os.path.join(folder_path, file)
                total_files += 1
                print(f"\nProcessing file: {file_path}")

                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    file_modified = False
                    
                    # 处理不同的数据结构
                    if isinstance(data, list):
                        items_to_process = data
                    elif isinstance(data, dict) and any(isinstance(data.get(k), list) for k in data):
                        # 如果是字典且包含列表，找到列表处理
                        for key, value in data.items():
                            if isinstance(value, list):
                                items_to_process = value
                                break
                    else:
                        continue

                    # 处理每个项目
                    for item in items_to_process:
                        if isinstance(item, dict) and 'question' in item:
                            original_question = item['question']
                            modified_question = process_sequence_in_question(original_question)
                            
                            # 只有当问题实际被修改时才计数
                            if original_question != modified_question:
                                item['question'] = modified_question
                                file_modified = True

                    # 只有当文件被修改时才写回
                    if file_modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        modified_count += 1
                        print(f"Modified and saved: {file_path}")
                    else:
                        print(f"No sequences to modify in: {file_path}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

    print(f"\nProcessing complete!")
    print(f"Total JSON files processed: {total_files}")
    print(f"Files modified: {modified_count}")

if __name__ == "__main__":
    directory_path = "/root/opencompass/Bio-Benchmark"
    process_json_files(directory_path)