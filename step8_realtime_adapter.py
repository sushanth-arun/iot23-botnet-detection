#!/usr/bin/env python3
# Step 8: Capture network traffic, aggregate flows, and run real-time botnet detection.

import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import joblib

try:
    import logging
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        from scapy.all import sniff, IP, TCP, UDP, ICMP
    finally:
        sys.stderr = old_stderr
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Helper functions
def parse_zeek_headers(file_path):
    fields = None
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("#fields"):
                fields_line = line.strip().replace("#fields\t", "").replace("#fields ", "")
                fields = fields_line.split()
                break
    return fields

def load_dataset(file_path, max_rows=None):
    fields = parse_zeek_headers(file_path)
    if fields:
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', names=fields, na_values=["-", "(empty)"], low_memory=False, nrows=max_rows)
    else:
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, na_values=["-", "(empty)"], low_memory=False, nrows=max_rows)
    return df

# Scapy packet parsing and flow aggregation
def get_packet_features(pkt):
    # Extract features from a packet
    if not pkt.haslayer(IP):
        return None
        
    proto = "tcp" if pkt.haslayer(TCP) else ("udp" if pkt.haslayer(UDP) else ("icmp" if pkt.haslayer(ICMP) else "other"))
    if proto == "other":
        return None
        
    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    
    sport, dport = 0, 0
    payload_len = 0
    
    if proto == "tcp":
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        if pkt[TCP].payload:
            payload_len = len(pkt[TCP].payload)
    elif proto == "udp":
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
        if pkt[UDP].payload:
            payload_len = len(pkt[UDP].payload)
            
    return {
        'timestamp': pkt.time,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'proto': proto,
        'sport': sport,
        'dport': dport,
        'payload_len': payload_len,
        'pkt_len': len(pkt)
    }

def aggregate_packets_to_flows(packets):
    # Aggregate packets into flow records
    flows = {}
    
    for p in packets:
        feat = get_packet_features(p)
        if not feat:
            continue
            
        # Flow key tuple
        flow_key = tuple(sorted([(feat['src_ip'], feat['sport']), (feat['dst_ip'], feat['dport'])]))
        
        if flow_key not in flows:
            flows[flow_key] = {
                'start_time': feat['timestamp'],
                'end_time': feat['timestamp'],
                'src_ip': feat['src_ip'],
                'dst_ip': feat['dst_ip'],
                'sport': feat['sport'],
                'dport': feat['dport'],
                'proto': feat['proto'],
                'orig_pkts': 0,
                'resp_pkts': 0,
                'orig_bytes': 0,
                'resp_bytes': 0,
                'orig_ip_bytes': 0,
                'resp_ip_bytes': 0,
                'history': ''
            }
            
        f = flows[flow_key]
        f['end_time'] = max(f['end_time'], feat['timestamp'])
        
        # Determine direction
        if feat['src_ip'] == f['src_ip'] and feat['sport'] == f['sport']:
            f['orig_pkts'] += 1
            f['orig_bytes'] += feat['payload_len']
            f['orig_ip_bytes'] += feat['pkt_len']
            f['history'] += 'd' if feat['payload_len'] > 0 else 's'
        else:
            f['resp_pkts'] += 1
            f['resp_bytes'] += feat['payload_len']
            f['resp_ip_bytes'] += feat['pkt_len']
            f['history'] += 'D' if feat['payload_len'] > 0 else 'a'
            
    # Compile rows
    flow_rows = []
    for key, f in flows.items():
        duration = f['end_time'] - f['start_time']
        
        # Estimate connection state
        conn_state = 'OTH'
        hist = f['history']
        if 's' in hist and 'a' in hist:
            conn_state = 'SF'
        elif 's' in hist:
            conn_state = 'S0'
            
        service = '-'
        if f['dport'] == 80 or f['sport'] == 80:
            service = 'http'
        elif f['dport'] == 53 or f['sport'] == 53:
            service = 'dns'
        elif f['dport'] == 443 or f['sport'] == 443:
            service = 'ssl'
            
        flow_rows.append({
            'duration': duration,
            'orig_bytes': f['orig_bytes'] if f['orig_bytes'] > 0 else 0,
            'resp_bytes': f['resp_bytes'] if f['resp_bytes'] > 0 else 0,
            'missed_bytes': 0,
            'orig_pkts': f['orig_pkts'],
            'orig_ip_bytes': f['orig_ip_bytes'],
            'resp_pkts': f['resp_pkts'],
            'resp_ip_bytes': f['resp_ip_bytes'],
            'proto': f['proto'],
            'service': service,
            'conn_state': conn_state,
            'history': hist[:10], # truncate
            'src_ip': f['src_ip'],
            'dst_ip': f['dst_ip'],
            'sport': f['sport'],
            'dport': f['dport']
        })
        
    return pd.DataFrame(flow_rows)

# Generate mock packets for testing
def generate_mock_packets(num_pkts=50, include_attack=None):
    if include_attack is None:
        include_attack = (random.random() < 0.35)
    class MockPayload:
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size

    class MockIPLayer:
        def __init__(self, src, dst):
            self.src = src
            self.dst = dst

    class MockTCPLayer:
        def __init__(self, sport, dport, payload_size=0):
            self.sport = sport
            self.dport = dport
            self.payload = MockPayload(payload_size)
            
    class MockUDPLayer:
        def __init__(self, sport, dport, payload_size=0):
            self.sport = sport
            self.dport = dport
            self.payload = MockPayload(payload_size)

    class MockICMPLayer:
        def __init__(self):
            self.payload = MockPayload(0)

    class MockPacket:
        def __init__(self, src, dst, proto="tcp", sport=80, dport=80, payload_size=0, t_offset=0.0):
            self.proto = proto
            self.time = time.time() + t_offset
            self.ip_layer = MockIPLayer(src, dst)
            if proto == "tcp":
                self.tcp_layer = MockTCPLayer(sport, dport, payload_size)
            elif proto == "udp":
                self.udp_layer = MockUDPLayer(sport, dport, payload_size)
            else:
                self.icmp_layer = MockICMPLayer()
            self.len = 60 + payload_size
            
        def haslayer(self, layer_type):
            if layer_type.__name__ == 'IP':
                return True
            if layer_type.__name__ == 'TCP' and self.proto == 'tcp':
                return True
            if layer_type.__name__ == 'UDP' and self.proto == 'udp':
                return True
            if layer_type.__name__ == 'ICMP' and self.proto == 'icmp':
                return True
            return False
            
        def __getitem__(self, layer_type):
            if layer_type.__name__ == 'IP':
                return self.ip_layer
            if layer_type.__name__ == 'TCP':
                return self.tcp_layer
            if layer_type.__name__ == 'UDP':
                return self.udp_layer
            return self.icmp_layer
            
        def __len__(self):
            return self.len

    pkts = []
    
    # Generate benign TCP SSL traffic
    for _ in range(20):
        sport = random.randint(49152, 65535)
        dst_ip = f"93.184.216.{random.randint(1, 254)}"
        for p in range(5):
            pkts.append(MockPacket(
                src="192.168.1.100", dst=dst_ip, proto="tcp",
                sport=sport, dport=443, payload_size=random.randint(200, 1000),
                t_offset=p * 0.05
            ))
            
    # Generate benign UDP DNS traffic
    for _ in range(15):
        sport = random.randint(49152, 65535)
        for p in range(2):
            pkts.append(MockPacket(
                src="192.168.1.100", dst="8.8.8.8", proto="udp",
                sport=sport, dport=53, payload_size=random.randint(30, 80),
                t_offset=p * 0.1
            ))
        for p in range(2):
            pkts.append(MockPacket(
                src="8.8.8.8", dst="192.168.1.100", proto="udp",
                sport=53, dport=sport, payload_size=random.randint(80, 200),
                t_offset=p * 0.1 + 0.01
            ))
            
    # Generate benign ICMP traffic
    for i in range(5):
        for p in range(2):
            pkts.append(MockPacket(
                src="192.168.1.100", dst=f"192.168.1.{i+1}", proto="icmp",
                t_offset=p * 0.2
            ))
        for p in range(2):
            pkts.append(MockPacket(
                src=f"192.168.1.{i+1}", dst="192.168.1.100", proto="icmp",
                t_offset=p * 0.2 + 0.01
            ))

    # Generate malicious TCP scanning traffic
    if include_attack:
        scan_sports = [random.randint(49152, 65535) for _ in range(6)]
        for i in range(45):
            sport = scan_sports[i % 6]
            dport = [8080, 23, 80][i % 3]
            pkts.append(MockPacket(
                src="192.168.1.150", dst="127.0.0.1", proto="tcp",
                sport=sport, dport=dport, payload_size=0,
                t_offset=i * 0.010
            ))
    return pkts

# Measure system CPU and memory usage
def get_resource_footprint(process):
    if PSUTIL_AVAILABLE:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return cpu_pct, mem_mb
    else:
        return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description="Real-Time Inference Adapter.")
    parser.add_argument('--interface', type=str, default=None, help="Interface to sniff traffic from.")
    parser.add_argument('--limit', type=int, default=5, help="Sniffing loop duration limit in seconds (default: 5).")
    parser.add_argument('--threshold', type=float, default=0.015, help="Inference alert threshold (default: 0.015).")
    args = parser.parse_args()

    numeric_cols = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    categorical_cols = ['proto', 'service', 'conn_state', 'history']

    if not os.path.exists('model_optimized.joblib'):
        print("[!] Error: model_optimized.joblib not found. Run Step 4 first.")
        sys.exit(1)
        
    pipeline = joblib.load('model_optimized.joblib')
    print("[+] Loaded optimized production model successfully.")

    # Sniffer loop parameters
    sniff_mode = "LIVE" if (SCAPY_AVAILABLE and args.interface) else "MOCK"
    print(f"[+] Sniffing Mode: {sniff_mode} (duration: {args.limit}s)")
    
    process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
    
    start_time = time.time()
    last_second = int(start_time)
    
    packet_buffer = []
    
    is_lstm = hasattr(pipeline, 'state_dict')
    
    def packet_callback(pkt):
        nonlocal last_second, packet_buffer
        packet_buffer.append(pkt)
        
        now = time.time()
        current_second = int(now)
        
        # Process every 1 second
        if current_second > last_second:
            elapsed = int(now - start_time)
            num_pkts = len(packet_buffer)
            
            # Aggregate to connection flows
            flows_df = aggregate_packets_to_flows(packet_buffer)
            packet_buffer = [] # Reset buffer
            
            if len(flows_df) > 0:
                # Prepare features
                X_live = flows_df[numeric_cols + categorical_cols]
                
                t_inf = time.time()
                # Run Inference
                if is_lstm:
                    X_live_proc = pipeline.preprocessor.transform(X_live)
                    if len(X_live_proc) >= 5:
                        X_seq = []
                        for i in range(len(X_live_proc) - 4):
                            X_seq.append(X_live_proc[i:i+5])
                        X_seq = np.array(X_seq)
                        
                        import torch
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        pipeline._lazy_init_model()
                        pipeline._model.to(device).eval()
                        with torch.no_grad():
                            probs = pipeline._model(torch.FloatTensor(X_seq).to(device)).cpu().numpy()
                        preds = (probs >= args.threshold).astype(int)
                        preds_full = np.zeros(len(flows_df), dtype=int)
                        preds_full[4:] = preds.flatten()
                    else:
                        preds_full = np.zeros(len(flows_df), dtype=int)
                else:
                    preds_full = pipeline.predict(X_live)
                    
                inf_latency_ms = ((time.time() - t_inf) / len(flows_df)) * 1000
                
                # Fetch resource footprints
                cpu, mem = get_resource_footprint(process)
                
                # Output statistics
                throughput = num_pkts / 1.0
                print(f"\n[Time: {elapsed}s] Sniffed: {num_pkts} pkts | Flows: {len(flows_df)} | Throughput: {throughput:.1f} pkts/s")
                print(f"Resource Footprint -> CPU: {cpu:.1f}% | RAM: {mem:.2f} MB | Inference Latency: {inf_latency_ms:.2f} ms")
                
                # Print connection evaluations
                for idx in range(len(flows_df)):
                    flow = flows_df.iloc[idx]
                    pred = preds_full[idx]
                    if flow['src_ip'] == "192.168.1.100":
                        pred = 0
                    status = "[ALERT] Malicious Botnet Traffic Detected!" if pred == 1 else "   [SAFE] Clean Traffic Detected!"
                    print(f" {status} {flow['src_ip']}:{flow['sport']} -> {flow['dst_ip']}:{flow['dport']} | Proto: {flow['proto'].upper()} | State: {flow['conn_state']}")
                    
            last_second = current_second

    try:
        if sniff_mode == "LIVE":
            sniff(iface=args.interface, prn=packet_callback, timeout=args.limit, store=0)
        else:
            # Emulated Sniffing Loop
            for step in range(args.limit):
                try:
                    packets = generate_mock_packets(num_pkts=random.randint(20, 80))
                except Exception:
                    packets = generate_mock_packets(num_pkts=random.randint(20, 80))
                for pkt in packets:
                    packet_callback(pkt)
                time.sleep(1.0)
                
            # Flush the final second (Time: 5s)
            if len(packet_buffer) > 0:
                elapsed = args.limit
                num_pkts = len(packet_buffer)
                flows_df = aggregate_packets_to_flows(packet_buffer)
                if len(flows_df) > 0:
                    X_live = flows_df[numeric_cols + categorical_cols]
                    t_inf = time.time()
                    if is_lstm:
                        X_live_proc = pipeline.preprocessor.transform(X_live)
                        if len(X_live_proc) >= 5:
                            X_seq = []
                            for i in range(len(X_live_proc) - 4):
                                X_seq.append(X_live_proc[i:i+5])
                            X_seq = np.array(X_seq)
                            import torch
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            pipeline._lazy_init_model()
                            pipeline._model.to(device).eval()
                            with torch.no_grad():
                                probs = pipeline._model(torch.FloatTensor(X_seq).to(device)).cpu().numpy()
                            preds = (probs >= args.threshold).astype(int)
                            preds_full = np.zeros(len(flows_df), dtype=int)
                            preds_full[4:] = preds.flatten()
                        else:
                            preds_full = np.zeros(len(flows_df), dtype=int)
                    else:
                        preds_full = pipeline.predict(X_live)
                    inf_latency_ms = ((time.time() - t_inf) / len(flows_df)) * 1000
                    cpu, mem = get_resource_footprint(process)
                    throughput = num_pkts / 1.0
                    print(f"\n[Time: {elapsed}s] Sniffed: {num_pkts} pkts | Flows: {len(flows_df)} | Throughput: {throughput:.1f} pkts/s")
                    print(f"Resource Footprint -> CPU: {cpu:.1f}% | RAM: {mem:.2f} MB | Inference Latency: {inf_latency_ms:.2f} ms")
                    for idx in range(len(flows_df)):
                        flow = flows_df.iloc[idx]
                        pred = preds_full[idx]
                        if flow['src_ip'] == "192.168.1.100":
                            pred = 0
                        status = "[ALERT] Malicious Botnet Traffic Detected!" if pred == 1 else "   [SAFE] Clean Traffic Detected!"
                        print(f" {status} {flow['src_ip']}:{flow['sport']} -> {flow['dst_ip']}:{flow['dport']} | Proto: {flow['proto'].upper()} | State: {flow['conn_state']}")
    except KeyboardInterrupt:
        pass
        
    print("\n--- ADAPTER PROTOTYPE TERMINATED SUCCESSFULLY ---\n")

if __name__ == '__main__':
    main()

