# === ROBOT CONTROL FUNCTION ===
def robot_control(actual_state, x,y):
    # Implement servo control logic here
    print("Executed robot_control with: (%d,%d)" %(x,y))
    updated_state = actual_state
    return updated_state

# === ROBOT CONTROL FUNCTION ===
def robot_state_init():
    # Implement robot state initialization logic here
    print("Executed robot state init")
    state_t0 = 0
    return state_t0