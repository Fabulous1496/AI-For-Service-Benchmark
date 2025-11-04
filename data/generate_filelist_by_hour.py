from datetime import datetime, timedelta

# 定义起始和结束时间
start_time_str = "11:10:00"
end_time_str = "12:10:00"
interval = timedelta(seconds=30)
date_format = "%H:%M:%S"

# 解析时间
# 我们使用一个固定的日期来计算时间差
dt_start = datetime.strptime(start_time_str, date_format)
dt_end = datetime.strptime(end_time_str, date_format)

# 文件名常量
prefix = "DAY1_A1_JAKE_"
suffix = ".mp4"
output_filename = "filelist.txt"

# 开始生成列表
current_dt = dt_start
with open(output_filename, 'w') as f:
    while current_dt < dt_end:
        # 格式化时间码为 HHMMSSTT (时分秒00)
        # 注意：脚本中最后两位 TT (帧数) 固定为 00，因为您的文件命名是 3000, 0000 交替。
        time_code = current_dt.strftime("%H%M%S") + "00" 
        
        # 拼接完整的文件名
        filename = f"{prefix}{time_code}{suffix}"
        
        # 写入 filelist.txt 格式
        f.write(f"file '{filename}'\n")
        
        # 递增 30 秒
        current_dt += interval

print(f"✅ {output_filename} 文件已成功生成！请检查文件内容是否正确。")