# Q-Learning Demo Project

ตัวอย่างโปรเจค Q-learning สำหรับใช้ประกอบการสอนในบทเรียน Reinforcement Learning
**🆕 อัปเดตใหม่: เพิ่มระดับความยากหลายแบบ!**

## ไฟล์ในโปรเจค

### 1. `simple_q_learning.py` (แนะนำสำหรับการสาธิต) ⭐
- ใช้ Python ธรรมดา ไม่ต้องติดตั้ง package เพิ่มเติม
- เหม## การขยายผลต่อ

สามารถขยายโปรเจคได้หลายทิศทาง:
- เพิ่ม different RL algorithms (SARSA, Double Q-Learning)
- ลองกับ environments อื่น (Frozen Lake, Cart Pole)
- **✅ เพิ่ม neural network function approximation (สำเร็จแล้ว!)**
- เพิ่ม advanced DQN features (Double DQN, Dueling DQN)
- เปรียบเทียบกับ other planning algorithms
- **สร้าง visualization ของ neural network learning**

### งานที่สามารถทำต่อ:
1. **Convolutional Neural Network** สำหรับ visual input
2. **Policy Gradient Methods** (REINFORCE, Actor-Critic)
3. **Multi-agent Reinforcement Learning**
4. **Real-world applications** (robotics, games)บการสาธิตในชั้นเรียน
- **🆕 รองรับ 4 ระดับความยาก: Easy, Normal, Hard, Maze**
- มี interactive mode ให้ปรับแต่งพารามิเตอร์ได้
- มีโหมดเปรียบเทียบความยากใหม่

### 2. `grid_world.py`
- Grid World environment พร้อม visualization
- ต้องใช้ numpy และ matplotlib

### 3. `q_learning.py`
- Q-Learning algorithm แบบเต็มรูปแบบ
- รองรับ epsilon decay และสถิติการเรียนรู้

### 4. `demo.py`
- Demo script รวม มี 3 โหมด
- ต้องติดตั้ง numpy และ matplotlib สำหรับ full version

### 5. `assignments.py`
- แบบฝึกหัดสำหรับนักเรียน
- หลายระดับความยาก

### 7. `pure_python_dqn.py` & `test_dqn.py` (ใหม่!) 🧠
- **Pure Python DQN**: Neural Network Function Approximation
- ไม่ต้องติดตั้ง numpy หรือ pytorch
- เปรียบเทียบ Tabular Q-Learning กับ Deep Q-Network
- แสดงการทำงานของ neural network แบบง่าย

### 8. `educational_comparison.py` (สำหรับการสอน!) 🎓
- สาธิตการเปรียบเทียบแบบละเอียด step-by-step
- อธิบายข้อดี/ข้อเสีย ของแต่ละแนวทาง
- แสดงการเรียนรู้แบบ learning curves
- เหมาะสำหรับใช้ในการสอน

### 8. `test_dqn.py` (ทดสอบ Neural Network)
- ทดสอบ Pure Python neural network implementation
- เปรียบเทียบผลลัพธ์ Tabular vs DQN

## ระดับความยากใหม่ 🎯

### 1. **Easy** (ง่าย)
- อุปสรรคน้อย เรียงกระจาย
- เหมาะสำหรับเริ่มต้นเรียนรู้
- Grid 4x4: 2 อุปสรรค
- Grid 5x5: 3 อุปสรรค

### 2. **Normal** (ปกติ)
- อุปสรรคปานกลาง มีกำแพงบางส่วน
- เหมาะสำหรับการเรียนรู้หลัก
- Grid 4x4: 2 อุปสรรค (แนวตั้ง)
- Grid 5x5: 4 อุปสรรค

### 3. **Hard** (ยาก)
- อุปสรรคมาก สร้างเส้นทางที่ซับซ้อน
- ต้องใช้กลยุทธ์ในการหาทาง
- Grid 4x4: 4 อุปสรรค
- Grid 5x5: 6 อุปสรรค

### 4. **Maze** (เขาวงกต) 🌀
- รูปแบบเขาวงกต ท้าทายที่สุด
- ต้องหาเส้นทางผ่านช่องแคบ
- สร้างกำแพงแบบกริด pattern

## การใช้งาน

### วิธีที่ 1: สาธิตแบบง่าย (แนะนำ) ⭐
```bash
cd q-learning-demo
python3 simple_q_learning.py
# เลือก 1 = Demo พร้อมเลือกระดับความยาก
```

### วิธีที่ 2: เปรียบเทียบความยาก 
```bash
python3 simple_q_learning.py
# เลือก 3 = เปรียบเทียบทุกระดับพร้อมกัน
```

### วิธีที่ 3: Interactive Mode
```bash
python3 simple_q_learning.py
# เลือก 2 = ปรับแต่งทุกพารามิเตอร์
```

### วิธีที่ 5: Neural Network Function Approximation 
```bash
python3 pure_python_dqn.py
# เปรียบเทียบ Tabular Q-Learning กับ Deep Q-Network
```

### วิธีที่ 6: ทดสอบ Neural Network แบบง่าย
```bash
python3 test_dqn.py  # ทดสอบ neural network implementation
```

## ตัวอย่างผลลัพธ์ระดับความยากต่างๆ

### Easy Level
```
Grid World 4x4 - easy
อุปสรรค: 2, เส้นทางสั้นที่สุด: 6 steps
Final reward: 9.20, Steps: 7
ประสิทธิภาพ: 85.7%
```

### Hard Level  
```
Grid World 4x4 - hard
อุปสรรค: 4, เส้นทางสั้นที่สุด: 6 steps
Final reward: 8.60, Steps: 12
ประสิทธิภาพ: 50.0%
```

### Deep Q-Network (DQN) Example
```
Pure Python DQN vs Tabular Q-Learning:
Grid World 3x3 - easy
Neural Network: Input=9, Hidden=32, Output=4

Tabular Q-Learning: 9.70 reward, 4 steps
Pure Python DQN:   -5.00 reward, 50 steps (needs more training)

→ ในปัญหาเล็ก Tabular มักจะเรียนรู้เร็วกว่า
→ DQN มีประโยชน์เมื่อ state space ใหญ่มาก
```

## 🎓 คำแนะนำสำหรับการสอน

### การนำเสนอในห้องเรียน:

1. **เริ่มด้วย Simple Demo** (15 นาที)
   ```bash
   python simple_q_learning.py
   ```
   - เลือก interactive mode
   - แสดงให้เห็น Q-table การเปลี่ยนแปลง
   - อธิบายคำศัพท์: state, action, reward, Q-value

2. **แสดงความซับซ้อน** (10 นาที)
   ```bash
   python simple_q_learning.py
   ```
   - เลือก difficulty levels: easy → normal → hard → maze
   - อธิบายว่า state space มีผลต่อ Q-table อย่างไร

3. **Neural Network vs Tabular** (20 นาที)
   ```bash
   python educational_comparison.py
   ```
   - อธิบายข้อจำกัดของ tabular approach
   - แสดง function approximation concept
   - เปรียบเทียบผลลัพธ์และความซับซ้อน

### การฝึกปฏิบัติ:

1. **แบบฝึกหัดพื้นฐาน**
   ```bash
   python assignments.py
   ```

2. **การทดลอง parameters**
   - ให้นักศึกษาแก้ไข learning_rate, epsilon, discount
   - สังเกตผลต่อการเรียนรู้

### คำถามสำหรับดึงความสนใจ:

- "ถ้า state space มี 1 ล้าน states จะเกิดอะไรขึ้น?"
- "ทำไม neural network ถึงช้ากว่าใน grid world เล็ก?"
- "เมื่อไหร่ควรใช้ function approximation?"

## 🎯 วิธีการใช้

### 1. **สถิติขั้นสูง**
- ประสิทธิภาพเทียบกับเส้นทางที่สั้นที่สุด
- อัตราความสำเร็จ (Success Rate)
- อัตราส่วนอุปสรรคต่อพื้นที่

### 2. **การปรับแต่งอัตโนมัติ**
- Episodes เพิ่มขึ้นสำหรับระดับยาก
- Epsilon เพิ่มขึ้นสำหรับ environment ซับซ้อน

### 4. **Neural Network Function Approximation** 🧠
- Pure Python implementation ของ Deep Q-Network
- เปรียบเทียบกับ Tabular Q-Learning
- แสดงข้อดี/ข้อเสียของแต่ละแนวทาง
- เหมาะสำหรับสอนความแตกต่างระหว่าง tabular กับ function approximation

## การใช้ในการสอน

### สำหรับการบรรยาย:
1. เริ่มด้วย Easy level เพื่อแสดงแนวคิด
2. เปลี่ยนเป็น Hard level เพื่อแสดงความท้าทาย
3. ใช้ Maze level เพื่อแสดงข้อจำกัด
4. **แสดง neural network function approximation**

### สำหรับ hands-on:
1. ให้นักเรียนทดลองทุกระดับ
2. เปรียบเทียบผลลัพธ์
3. วิเคราะห์สาเหตุของความแตกต่าง
4. **ทดลองเปรียบเทียบ Tabular vs DQN**

### สำหรับการมอบหมาย:
1. ให้สร้างระดับความยากของตัวเอง
2. ทดลองปรับพารามิเตอร์สำหรับแต่ละระดับ
3. วิเคราะห์ผลกระทบของความซับซ้อน
4. **เปรียบเทียบ tabular กับ function approximation**

## พารามิเตอร์ที่แนะนำสำหรับแต่ละระดับ

| ระดับ | Episodes | Epsilon | Learning Rate | หมายเหตุ |
|-------|----------|---------|---------------|----------|
| Easy | 800 | 0.2 | 0.1 | เรียนรู้เร็ว |
| Normal | 1000 | 0.3 | 0.1 | ค่าปกติ |
| Hard | 1500 | 0.4 | 0.1 | ต้องการ exploration มาก |
| Maze | 2000 | 0.5 | 0.05 | เรียนรู้ช้าแต่แน่นอน |

## Neural Network Function Approximation 🧠

### Pure Python Implementation:
- **SimpleMLP**: Multi-layer perceptron ใช้ pure Python
- **PurePythonDQN**: Deep Q-Network implementation
- **Target Network**: Stable learning เหมือน DQN paper
- **Experience Replay**: เก็บ transitions และ sample แบบสุ่ม

### ความแตกต่างระหว่าง Tabular กับ Function Approximation:

| แนวทาง | ข้อดี | ข้อเสีย | เหมาะกับ |
|--------|-------|---------|----------|
| **Tabular Q-Learning** | เรียนรู้เร็ว, แน่นอน | จำกัดด้วย state space | ปัญหาเล็ก |
| **Neural Network DQN** | รองรับ state ใหญ่, generalize ได้ | เรียนรู้ช้า, ไม่เสถียร | ปัญหาใหญ่ |

### ตัวอย่างการใช้งาน:
```python
# Tabular Q-Learning
tabular_agent = SimpleQLearning(n_states=16, n_actions=4)

# Neural Network DQN  
dqn_agent = PurePythonDQN(state_size=16, action_size=4)
```

### การสอนในชั้นเรียน:
1. **เริ่มจาก Tabular**: ง่าย เข้าใจได้
2. **อธิบายข้อจำกัด**: เมื่อ state space ใหญ่
3. **แนะนำ Neural Network**: วิธีแก้ปัญหา
4. **เปรียบเทียบผลลัพธ์**: ดูข้อดี/ข้อเสีย

## ข้อดีของการเพิ่มระดับความยาก

1. **เสริมความเข้าใจ**: เห็นผลกระทบของความซับซ้อน
2. **เปรียบเทียบได้**: เห็นข้อจำกัดของ algorithm
3. **ความน่าสนใจ**: เพิ่มความท้าทายให้นักเรียน
4. **ความเป็นจริง**: สะท้อนปัญหาโลกจริง

---

*โปรเจคนี้ได้รับการอัปเดตให้มีความซับซ้อนและความน่าสนใจมากขึ้น เหมาะสำหรับการสอน Q-learning ในระดับต่างๆ*

## การใช้งาน

### วิธีที่ 1: สาธิตแบบง่าย (แนะนำ)
```bash
cd q-learning-demo
python simple_q_learning.py
```

### วิธีที่ 2: Full version (ต้องติดตั้ง packages)
```bash
pip install numpy matplotlib
python demo.py
```

## ฟีเจอร์หลัก

### 1. Grid World Environment
- ขนาดปรับได้ (3x3 ถึง 8x8)
- มีจุดเริ่มต้น (S), เป้าหมาย (G), และอุปสรรค (X)
- Reward system: Goal=+10, Obstacle=-5, Step=-0.1

### 2. Q-Learning Algorithm
- Epsilon-greedy exploration
- Temporal difference learning
- Q-value updates ตาม Bellman equation

### 3. Visualizations
- Grid world map
- Learning curves
- Optimal policy
- Q-table values

### 4. Interactive Features
- ปรับแต่ง learning rate, epsilon, episodes
- แสดงเส้นทางการเดิน step-by-step
- เปรียบเทียบผลก่อนและหลังการฝึก

## พารามิเตอร์สำคัญ

- **Learning Rate (α)**: 0.1 (ควบคุมความเร็วการเรียนรู้)
- **Discount Factor (γ)**: 0.9-0.95 (ความสำคัญของ future rewards)
- **Epsilon (ε)**: 0.1-0.2 (อัตราการสำรวจ)
- **Episodes**: 1000+ (จำนวนรอบการฝึก)

## การใช้ในการสอน

### สำหรับการบรรยาย:
1. เริ่มด้วย `simple_q_learning.py` โหมด demo
2. อธิบาย concepts พร้อมดูผล Q-table
3. แสดง learning process แบบ step-by-step

### สำหรับ hands-on:
1. ให้นักเรียนรัน interactive mode
2. ปรับแต่งพารามิเตอร์และสังเกตผล
3. เปรียบเทียบ performance ต่างๆ

### สำหรับการมอบหมาย:
1. ให้ดัดแปลง reward structure
2. เพิ่มอุปสรรคหรือเปลี่ยน grid size
3. ทดลอง different exploration strategies

## ตัวอย่างผลลัพธ์

```
Episode 0: Average reward = -2.50, Epsilon = 0.200
Episode 200: Average reward = 5.20, Epsilon = 0.190
Episode 400: Average reward = 7.80, Epsilon = 0.181
Episode 600: Average reward = 8.90, Epsilon = 0.172
Episode 800: Average reward = 9.10, Epsilon = 0.164

Final performance: Reward=9.90, Steps=8
```

## Q-Table ตัวอย่าง

```
State |   ↑   |   ↓   |   ←   |   →   
----------------------------------------
    0 | -0.50 |  1.20 | -0.30 |  2.10
    1 |  0.80 |  2.50 | -0.10 |  3.40
    2 |  1.90 |  4.20 |  0.60 |  5.80
   ...
```



## การขยายผลต่อ

สามารถขยายโปรเจคได้หลายทิศทาง:
- เพิ่ม different RL algorithms (SARSA, Double Q-Learning)
- ลองกับ environments อื่น (Frozen Lake, Cart Pole)
- เพิ่ม neural network function approximation
- เปรียบเทียบกับ other planning algorithms

---

*โปรเจคนี้เหมาะสำหรับการสอน Q-learning ในระดับปริญญาตรี *
