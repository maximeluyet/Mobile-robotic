import asyncio
from tdmclient import ClientAsync, aw
import threading
import time
import random

# TODO: document

class ThymioClientThreaded:
    def __init__(self):
        self.client = ClientAsync()
        self.node = None

        self.prox = list()
        self.ground = list()
        self.vleft = 0
        self.vright = 0

        self.vleft_set = 0
        self.vright_set = 0

        self.prox_changed = False
        self.ground_changed = False

        self.running = True

        threading.Thread(target=self.run_thread, args=(), daemon=True).start()

    
    def run_thread(self):
        asyncio.run(self.async_loop())

    
    async def async_loop(self):

        print("Thymio thread started")

        self.node = await self.client.wait_for_node()
        await self.node.lock()
        await self.node.compile("")

        print("Connected to Thymio")

        await self.node.watch(variables=True)

        self.node.add_variables_changed_listener(self.on_vars)

        # Keep the thread going
        while self.running:
            v = {"motor.left.speed": [100], "motor.right.speed": [100]}
            await self.node.set_variables(v)
            print(f"Set speeds: {self.vleft_set}, {self.vright_set}")
            await self.client.sleep(0.2)

        v = {"motor.left.speed": [0], "motor.right.speed": [0]}
        await self.node.set_variables(v)

        print("Thymio thread stopped, unlocking robot")
        await self.node.unlock()

    
    def on_vars(self, node, variables):

        if "prox.horizontal" in variables:
            self.prox = variables["prox.horizontal"]
            self.prox_changed = True

        if "prox.ground.reflected" in variables:
            self.ground = variables["prox.ground.reflected"]
            self.ground_changed = True

        if "motor.left.speed" in variables:
            self.vleft = variables["motor.left.speed"]

        if "motor.right.speed" in variables:
            self.vright = variables["motor.right.speed"]


    def stop(self):
        self.running = False

    def set_speed(self, vleft, vright):
        self.vleft_set = vleft
        self.vright_set = vright

    def get_prox(self):
        return self.prox
    
    def get_ground(self):
        return self.ground
    
    def has_prox_changed(self):
        changed = self.prox_changed
        self.prox_changed = False
        return changed
    
    def has_ground_changed(self):
        changed = self.ground_changed
        self.ground_changed = False
        return changed
    
    def get_speed(self):
        return self.vleft, self.vright


class ThymioClient:
    def __init__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())

        aw(self.node.compile(""))

        print("Client created and connected to Thymio.")


    def close(self):
        print("Stopping Thymio client and unlocking robot.")
        aw(self.node.unlock())

    def set_motors(self, left_speed, right_speed):
        v = {
            "motor.left.target": [left_speed],
            "motor.right.target": [right_speed],
        }
        aw(self.node.set_variables(v))

    def get_measurements(self):
        aw(self.node.wait_for_variables({"prox.horizontal", "motor.right.speed", "prox.ground.reflected"}))
        prox = list(self.node.v.prox.horizontal)
        v_l = self.node.v.motor.left.speed
        v_r = self.node.v.motor.right.speed
        ground = list(self.node.v.prox.ground.reflected)
        return prox, v_l, v_r, ground
    
    def wait(self, duration):
        aw(self.client.sleep(duration))

# client = ThymioClient()
# client.set_motors(200, 200)

# start = time.time()

# for i in range(20):
#     frame = time.time()
#     prox, v_l, v_r, ground = client.get_measurements()
#     print(f"Time: {time.time() - start:.2f}s")
#     print(f"Prox: {prox}, V_L: {v_l}, V_R: {v_r}, Ground: {ground}")

#     client.wait(0.1)


# client.set_motors(0, 0)