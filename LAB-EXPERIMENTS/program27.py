def intersection_controller(car_state):
    # car_state: (traffic_light_color, cross_traffic_approaching)
    light, cross = car_state
    if light == 'RED' or cross:
        return 'STOP'
    elif light == 'YELLOW':
        return 'DECELERATE'
    return 'PROCEED'

scenarios = [('GREEN', False), ('GREEN', True), ('RED', False), ('YELLOW', False)]
for sc in scenarios:
    print(f"Scenario {sc} -> Decision: {intersection_controller(sc)}")