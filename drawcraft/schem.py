import mcschematic
import numpy as np

'''
piston_top_sticky = sticky_piston[facing={}].format(kierunek(way))
piston_top = piston[facing={}].format(kierunek(way))

chiseled_bookshelf_occupied = chiseled_bookshelf[slot_0_occupied=true,slot_1_occupied=true,slot_2_occupied=true,slot_3_occupied=true,
slot_4_occupied=true,slot_5_occupied=true, facing={}].format(kierunek(way))

'''

way = "north"

def kierunek(looking_at, rotation):
    kierunki = {"north" : 0, "east" : 1, "south" : 2, "west" : 3}
    rotacja = abs(kierunki[looking_at] + rotation)
    if rotacja > 4:
        rotacja -= 4
    return list(kierunki.keys())[rotacja]

def blockDataHandler(name):
    patterns_to_remove = ["_on.png", "_off.png", ".png", "_front", "_top", "_side0", "_side1", "_side2", "_side3", "_bottom", "_side", "_vertical", "_0", "_1", "_2", "_3", "_empty", "_back"]
    patterns_to_add = {"dried_kelp":"_block"}
    patterns_to_change = {"piston_sticky": "sticky_piston", "chiseled_bookshelf_occupied":"chiseled_bookshelf[slot_0_occupied=true,slot_1_occupied=true,slot_2_occupied=true,slot_3_occupied=true,slot_4_occupied=true,slot_5_occupied=true]"}


    for pattern in patterns_to_remove:
        if pattern in name:
            name = name.replace(pattern, "")
    for pattern in patterns_to_add:
        if name == pattern:
            name = name+patterns_to_add[pattern]
    for pattern in patterns_to_change:
        if name == pattern:
            name = patterns_to_change[pattern]
        
    return "minecraft:"+name


def createScheamtic(dir, m, name):
    print("Generating schematic...")
    # mirror and rotate matrix 90 degrees
    matrix = np.fliplr(m)
    matrix = np.rot90(m, 3)

    # Create a new schematic object
    schem = mcschematic.MCSchematic()
    # Iterate through the matrix
    for x, row in enumerate(matrix):
        for y, block in enumerate(row):
            schem.setBlock((x, y, 0), blockDataHandler(block)) # update rotation check and block metadata
            '''
            minecraft:oak_log[axis=z] is an oak_log facing the z axis.
            chest[]{Items:[{Slot:0b, Count:1b, id:"minecraft:redstone"}]} is a chest with its first slot filled with one redstone dust.
            So if we want to place a rightside up stone_slab in our schematic, the blockData used would be stone_slab[type=top] just like in Minecraft! 
            '''

    print("Created schematic.")
    # Save the schematic
    schem.save(dir, name, mcschematic.Version.JE_1_20_1)
