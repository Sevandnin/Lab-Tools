import os
from openpyxl import Workbook

# CRC32 校验函数（移植自已有实现）
CRC32_POLYNOMIAL = 0xEDB88320

def calc_crc32_value(value):
    ulCRC = value
    for i in range(8):
        if ulCRC & 1:
            ulCRC = ((ulCRC >> 1) ^ CRC32_POLYNOMIAL) & 0xFFFFFFFF
        else:
            ulCRC = (ulCRC >> 1) & 0xFFFFFFFF
    return ulCRC

def calc_block_crc32(data_bytes):
    ulCRC = 0
    for b in data_bytes:
        ulTmp1 = ((ulCRC >> 8) & 0x00FFFFFF) & 0xFFFFFFFF
        ulTmp2 = calc_crc32_value((ulCRC ^ b) & 0xFF)
        ulCRC = (ulTmp1 ^ ulTmp2) & 0xFFFFFFFF
    return ulCRC

def verify_crc32(line: str) -> bool:
    try:
        if '*' not in line:
            return False
        parts = line.split('*')
        if len(parts) < 2:
            return False
        data_part = parts[0]
        checksum_str = parts[1].strip().split()[0] if parts[1].strip() else ""
        if len(checksum_str) != 8:
            return False
        expected_crc = int(checksum_str, 16)
        if not data_part.startswith('#'):
            return False
        data_to_check = data_part[1:]
        data_bytes = data_to_check.encode('ascii', errors='ignore')
        calculated = calc_block_crc32(data_bytes)
        return calculated == expected_crc
    except Exception:
        return False

def parse_bestposa_line(line: str):
    try:
        header, body = line.split(';', 1)
        header_parts = header.split(',')
        week_sec = float(header_parts[6])
        parts = body.split(',')
        lat = float(parts[2])
        lon = float(parts[3])
        hgt = float(parts[4])
        undulation = float(parts[5])
        corrected_hgt = hgt + undulation
        return week_sec, lat, lon, corrected_hgt
    except Exception:
        return None

def parse_corrimudata_line(line: str):
    try:
        if not verify_crc32(line):
            return None
        if ';' not in line:
            return None
        data_part = line.split(';', 1)[1]
        if '*' in data_part:
            data_part = data_part.split('*', 1)[0]
        fields = data_part.split(',')
        if len(fields) < 8:
            return None
        ts = float(fields[1])
        angle_x = float(fields[2])
        angle_y = float(fields[3])
        angle_z = float(fields[4])
        vel_x = float(fields[5])
        vel_y = float(fields[6])
        vel_z = float(fields[7])
        return ts, angle_x, angle_y, angle_z, vel_x, vel_y, vel_z
    except Exception:
        return None

def extract_bestposa_and_corrimu(dat_path: str, out_dir: str = None):
    if not os.path.exists(dat_path):
        raise FileNotFoundError(dat_path)
    rtk_rows = []
    imu_rows = []
    with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#BESTPOSA'):
                r = parse_bestposa_line(line)
                if r:
                    rtk_rows.append(r)
            elif 'CORRIMUDATA' in line:
                r = parse_corrimudata_line(line)
                if r:
                    imu_rows.append(r)

    base_dir = out_dir or os.path.dirname(dat_path) or '.'
    imu_xlsx = os.path.join(base_dir, 'IMU_Data.xlsx')
    rtk_xlsx = os.path.join(base_dir, 'RTK_Data.xlsx')

    wb_imu = Workbook()
    ws_imu = wb_imu.active
    for row in imu_rows:
        ws_imu.append(list(row))
    wb_imu.save(imu_xlsx)

    wb_rtk = Workbook()
    ws_rtk = wb_rtk.active
    for row in rtk_rows:
        ws_rtk.append(list(row))
    wb_rtk.save(rtk_xlsx)

    return {'rtk_count': len(rtk_rows), 'imu_count': len(imu_rows), 'rtk_xlsx': rtk_xlsx, 'imu_xlsx': imu_xlsx}

if __name__ == '__main__':
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title='选择 .dat 文件', filetypes=[('DAT files','*.dat'), ('All files','*.*')])
        root.destroy()
    except Exception:
        path = ''

    if not path:
        path = input('输入 .dat 文件路径（回车使用 UART-COM7-921600.dat）: ').strip() or 'UART-COM7-921600.dat'

    result = extract_bestposa_and_corrimu(path)
    print('解析完成：')
    print(f"RTK 行数: {result['rtk_count']}, 输出: {result['rtk_xlsx']}")
    print(f"IMU 行数: {result['imu_count']}, 输出: {result['imu_xlsx']}")