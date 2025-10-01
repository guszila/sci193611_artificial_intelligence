from pacman_module.game import Agent
from pacman_module.pacman import Directions
import heapq
import itertools

# ---------- Utilities ----------

def state_key(state):
    """
    ทำ key ของสถานะให้แฮชได้ชัวร์:
    - ตำแหน่งแพคแมน (x, y)
    - เซ็ตอาหารที่เหลือ (tuple ของตำแหน่งที่ sort แล้ว)
    - แคปซูลที่เหลือ (tuple ของตำแหน่งที่ sort แล้ว)
    """
    pos = state.getPacmanPosition()
    food = tuple(sorted(state.getFood().asList()))
    capsules = tuple(sorted(state.getCapsules()))
    return (pos, food, capsules)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def max_manhattan_to_food(pos, foods):
    """
    Heuristic แบบ admissible: ระยะแมนฮัตตันไปยัง 'อาหารที่ไกลที่สุด'
    - เป็น lower bound ที่ตึงกว่าใช้ 'อาหารที่ใกล้ที่สุด'
    - ไม่พึ่งพาโครงสร้างเขาวงกต (แอดมิท)
    """
    if not foods:
        return 0
    return max(manhattan(pos, f) for f in foods)

# ---------- A* Agent ----------

class PacmanAgent(Agent):
    """
    A* search สำหรับเก็บอาหารให้หมดกระดาน
    - ใช้ priority queue (heap) โดย key = f = g + h
    - เก็บ best_g[state_key] เพื่อ reopen โหนดเมื่อพบเส้นทางดีกว่า
    """
    def __init__(self, args):
        self.moves = []

    def get_action(self, state):
        if not self.moves:
            self.moves = self.astar(state) or []
        return self.moves.pop(0) if self.moves else Directions.STOP

    def astar(self, start_state):
        # เตรียมข้อมูลเริ่มต้น
        start_pos = start_state.getPacmanPosition()
        start_foods = start_state.getFood().asList()
        start_h = max_manhattan_to_food(start_pos, start_foods)

        # priority queue: (f, g, tie, state, path)
        heap = []
        counter = itertools.count()
        heapq.heappush(heap, (start_h, 0, next(counter), start_state, []))

        # บันทึกระยะทางดีที่สุดที่รู้ต่อ state แต่ละตัว
        best_g = {}

        while heap:
            f, g, _, current, path = heapq.heappop(heap)

            # ถ้าชนะแล้วก็จบ
            if current.isWin():
                return path

            ck = state_key(current)

            # ถ้าเราเคยเห็น state นี้ด้วย g ที่ดีกว่าแล้ว ให้ข้าม
            if g > best_g.get(ck, float('inf')):
                continue

            # อัปเดต g ที่ดีที่สุดของสถานะนี้
            best_g[ck] = g

            # ขยายเพื่อนบ้าน
            for next_state, action in current.generatePacmanSuccessors():
                new_path = path + [action]
                new_g = g + 1

                # คำนวณ heuristic ใหม่
                npos = next_state.getPacmanPosition()
                nfoods = next_state.getFood().asList()
                h = max_manhattan_to_food(npos, nfoods)

                nk = state_key(next_state)

                # ถ้าพบเส้นทางสั้นกว่าให้ push และบันทึก g ใหม่
                if new_g < best_g.get(nk, float('inf')):
                    best_g[nk] = new_g
                    heapq.heappush(heap, (new_g + h, new_g, next(counter), next_state, new_path))

        # หาไม่เจอ (ไม่น่าจะเกิดในเลย์เอาต์ปกติ)
        return []
