from urdfpy import URDF

robot = URDF.load('/home/csl/legged_rl/legged_gym/resources/robots/anymal_b/urdf/anymal_b.urdf')

def print_tree(link, indent=0):
    print(' ' * indent + f'- {link.name}')
    for joint in robot.joints:
        if joint.parent == link.name:
            child = robot.link_map[joint.child]
            print_tree(child, indent + 2)

# Start from the root link
print_tree(robot.base_link)
