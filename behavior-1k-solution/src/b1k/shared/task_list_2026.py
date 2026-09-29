"""2026 BEHAVIOR Challenge task list (index == challenge task id == 2026-challenge-demos task_index).

Generated from BEHAVIOR-1K docs/challenge/task_data.json, cross-checked against
datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv. Tasks 0-49 are the 2025 challenge tasks
with unchanged ids. INSTRUCTIONS maps the natural-language instruction (used as the task string in some
LeRobot exports) back to the task id.
"""

TASK_NAMES_2026 = (
    "turning_on_radio",  # 0
    "picking_up_trash",  # 1
    "putting_away_Halloween_decorations",  # 2
    "cleaning_up_plates_and_food",  # 3
    "can_meat",  # 4
    "setting_mousetraps",  # 5
    "hiding_Easter_eggs",  # 6
    "picking_up_toys",  # 7
    "rearranging_kitchen_furniture",  # 8
    "putting_up_Christmas_decorations_inside",  # 9
    "set_up_a_coffee_station_in_your_kitchen",  # 10
    "putting_dishes_away_after_cleaning",  # 11
    "preparing_lunch_box",  # 12
    "loading_the_car",  # 13
    "carrying_in_groceries",  # 14
    "bringing_in_wood",  # 15
    "moving_boxes_to_storage",  # 16
    "bringing_water",  # 17
    "tidying_bedroom",  # 18
    "outfit_a_basic_toolbox",  # 19
    "sorting_vegetables",  # 20
    "collecting_childrens_toys",  # 21
    "putting_shoes_on_rack",  # 22
    "boxing_books_up_for_storage",  # 23
    "storing_food",  # 24
    "clearing_food_from_table_into_fridge",  # 25
    "assembling_gift_baskets",  # 26
    "sorting_household_items",  # 27
    "getting_organized_for_work",  # 28
    "clean_up_your_desk",  # 29
    "setting_the_fire",  # 30
    "clean_boxing_gloves",  # 31
    "wash_a_baseball_cap",  # 32
    "wash_dog_toys",  # 33
    "hanging_pictures",  # 34
    "attach_a_camera_to_a_tripod",  # 35
    "clean_a_patio",  # 36
    "clean_a_trumpet",  # 37
    "spraying_for_bugs",  # 38
    "spraying_fruit_trees",  # 39
    "make_microwave_popcorn",  # 40
    "cook_cabbage",  # 41
    "chop_an_onion",  # 42
    "slicing_vegetables",  # 43
    "chopping_wood",  # 44
    "cook_hot_dogs",  # 45
    "cook_bacon",  # 46
    "freeze_pies",  # 47
    "canning_food",  # 48
    "make_pizza",  # 49
    "freeze_fruit",  # 50
    "cook_a_brisket",  # 51
    "sorting_bottles_cans_and_paper",  # 52
    "tidying_living_room",  # 53
    "putting_away_toys",  # 54
    "re_shelving_library_books",  # 55
    "make_rose_centerpieces",  # 56
    "sweeping_garage",  # 57
    "stacking_wood",  # 58
    "organizing_art_supplies",  # 59
    "scrubbing_bathroom_floor",  # 60
    "bringing_paper_to_recycling",  # 61
    "halve_an_egg",  # 62
    "installing_smoke_detectors",  # 63
    "setting_the_table",  # 64
    "unloading_the_car",  # 65
    "turning_out_all_lights_before_sleep",  # 66
    "boxing_food_after_dinner",  # 67
    "cleaning_up_branches_and_twigs",  # 68
    "vacuuming_floors",  # 69
    "thawing_frozen_food",  # 70
    "clean_your_rusty_garden_tools",  # 71
    "cook_a_frozen_pie",  # 72
    "organizing_school_stuff",  # 73
    "carrying_out_garden_furniture",  # 74
    "put_together_a_basic_pruning_kit",  # 75
    "dispose_of_glass",  # 76
    "installing_a_modem",  # 77
    "make_cabinet_doors",  # 78
    "polishing_shoes",  # 79
    "clean_up_broken_glass",  # 80
    "packing_meal_for_delivery",  # 81
    "store_batteries",  # 82
    "store_honey",  # 83
    "tidying_bathroom",  # 84
    "putting_dirty_dishes_in_sink",  # 85
    "make_gift_bags_for_baby_showers",  # 86
    "collecting_aluminum_cans",  # 87
    "rearrange_your_room",  # 88
    "installing_a_fax_machine",  # 89
    "composting_waste",  # 90
    "store_produce",  # 91
    "installing_a_scanner",  # 92
    "clean_a_keyboard",  # 93
    "dispose_of_batteries",  # 94
    "cook_brussels_sprouts",  # 95
    "cook_broccolini",  # 96
    "setup_a_bar_for_a_cocktail_party",  # 97
    "laying_tile_floors",  # 98
    "sorting_books_on_shelf",  # 99
)

TASK_NAME_TO_ID = {name: i for i, name in enumerate(TASK_NAMES_2026)}

INSTRUCTIONS = {
    "Turn on the radio receiver that's on the table in the living room.": 0,
    "Put the three cans of soda from the living room inside the trash can in the kitchen.": 1,
    "Place each of the two pumpkins and all three candles from the living room inside a cabinet in the living room (use any cabinet), then make sure every cabinet is closed, and position the cauldron so it is next to a table in the living room.": 2,
    "From the breakfast table in the kitchen, move both pizzas - keeping each on its plate - into the same refrigerator, put both bowls into one sink, and make sure the refrigerator is closed.": 3,
    "Open the kitchen cabinet, take out the two hinged jars, open them, place exactly two cooked bratwursts from the chopping board on the countertop into each jar, then close both jars, put them back inside the cabinet, and close the cabinet.": 4,
    "Take the four mousetraps from the cabinet in the bathroom and place them on the bathroom floor. Make sure all four end up on the same floor surface, and ensure that at least two of them are either under or directly next to the same bathroom sink.": 5,
    "Take the three Easter eggs out of the wicker basket on the lawn in the garden, then place them on the lawn next to a single tree (choose any tree) so that all three eggs are next to the same tree and none are left in the basket.": 6,
    "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.": 7,
    "Move the toaster, food processor, and French press from the kitchen countertop into the same kitchen cabinet, and make sure that cabinet is closed at the end.": 8,
    "In the living room, take the wreath, three candy canes, and two pillar candles out of the wicker basket. Place the wreath and two of the candy canes on the same living-room sofa. Put the remaining candy cane on top of a dining-room table. Put both pillar candles together on top of one dining-room table (they can share the same table). Finally, place all three gift boxes under or right next to the Christmas tree in the living room.": 9,
    "Set up a coffee station on the kitchen countertop: keep the coffee maker on the countertop, move the bottle of coffee from the kitchen shelf to the counter next to the coffee maker, place a paper coffee filter on top of the coffee maker, put the saucer next to the coffee maker with the coffee cup on the saucer, and place the electric kettle next to the coffee maker.": 10,
    "In the kitchen, gather all eight plates from the two countertops, place them all inside a single cabinet (either one), and make sure all cabinets are closed when you're done.": 11,
    "Put both apple halves, the club sandwich, and the chocolate chip cookie from the chopping board on the kitchen countertop into the packing box on the countertop. Then take the bottle of tea out of the refrigerator, put it into the same box, and close the refrigerator when you're done.": 12,
    "Put the digital camera from the living room table into the container on the living room floor. Then take the container and the tennis racket to the garage, place both in the car trunk, and close it.": 13,
    "Take the sack of groceries out of the car trunk in the garage, bring it to the kitchen, and put both the tomato and the carton of milk into the refrigerator in the kitchen. When you're done, close the car trunk and make sure the refrigerator in the kitchen is closed.": 14,
    "Bring the three plywood sheets from the garden into the corridor and place them on the floor there.": 15,
    "Move the two storage containers from the living room to the garage. In the garage, place one container on the floor and stack the other container on top of it (either order is fine).": 16,
    "Retrieve the two bottles from the refrigerator in the kitchen, bring them to the living room, and place both on the coffee table. Make sure the refrigerator is closed when you finish.": 17,
    "In the bedroom, move the book from the bed onto either nightstand, and place the two sandals side by side next to the bed.": 18,
    "In the utility room, put the drill, pliers, flashlight, Allen wrench, and screwdriver from the tabletop into the toolbox, keep the toolbox on the tabletop, and close the toolbox.": 19,
    "Sort the vegetables from the two wicker baskets on the kitchen floor into the mixing bowls on the kitchen countertop: put all three bok choy and all three Vidalia onions together into one mixing bowl; put both leeks and both broccoli together into a second mixing bowl; and put all three sweet corn into a third mixing bowl.": 20,
    "Pick up the two dice from the bed, the two teddy bears from the floor, and the two board games (one from the desk and one from the bed), and place them all inside the same bookcase in the child's room.": 21,
    "Pick up the two gym shoes and the two sandals from the corridor floor and place them onto the hallstand (shoe rack) in the corridor, making sure they are on the rack and not on the floor. Arrange them so the two gym shoes are next to each other and the two sandals are next to each other.": 22,
    "Put all six books from the bookcases in the living room into the box on the living room floor.": 23,
    "Put away all the food on the kitchen countertop by storing it inside the kitchen cabinets: move the two boxes of oatmeal, two bags of chips, two bottles of olive oil, and two jars of sugar from the countertop into the kitchen cabinets (each item can go into either cabinet).": 24,
    "Pack the half chicken and the half apple pie from the plates on the breakfast table into the two tupperware containers from the countertop, then put both tupperware containers inside the refrigerator in the kitchen and make sure the refrigerator is closed at the end.": 25,
    "Place one candle, one butter cookie, one piece of Swiss cheese, and one bow from the table into each of the four wicker baskets on the floor in the living room.": 26,
    "From the two baskets on the bedroom floor, take out the items and organize them in the bathroom: place both detergent bottles under the bathroom sink next to each other; put the box of sanitary napkins on the bathroom shelf; set the soap dispenser on the sink; make sure the cup remains on the sink; put both the toothpaste tube and the toothbrush inside the cup.": 27,
    "In the bedroom, organize the workspace by keeping the computer under the desk, ensuring the monitor is on the desk, placing the keyboard on the desk next to the monitor, placing the mouse on the desk next to the keyboard, moving the folder from the swivel chair onto the desk next to the mouse, stacking the notebook on top of the folder with the pen on top of the notebook, and positioning the swivel chair next to the desk.": 28,
    "In the child's room, clean up the desk: put both folders and both paperback books into the bookcase; put the pencil and both pens into the pencil case and leave the case on the desk; take the stapler out of the bookcase and place it on the desk; move the laptop from the bed onto the desk and close it.": 29,
    "In the living room, place the newspaper from the table into the wood fireplace, then put both pieces of firewood from the floor on top of the newspaper. Use the cigar lighter to ignite any one of the items so that the whole pile catches fire, and then turn the lighter off.": 30,
    "Wash the two dusty boxing gloves from the countertop in the utility room in the washer until they are no longer covered with dust.": 31,
    "Wash the two baseball caps on the countertop in the utility room using the washer until they are no longer dirty.": 32,
    "In the utility room, take the two teddy toys, the tennis ball, and the softball out of the cabinet and wash them in the washer so that both teddies are free of dirt and dust, the tennis ball has no debris, and the softball has no dirt.": 33,
    "Pick up the poster from the kitchen countertop and hang it on one of the wall nails in the kitchen.": 34,
    "Attach the digital camera to the camera tripod in the bedroom.": 35,
    "Pick up the broom in the garden and sweep the mud off the patio floor until the floor is no longer covered in mud.": 36,
    "In the bedroom, pick up the scrub brush from the desk and scrub the cornet (trumpet) on the desk until it's no longer covered in dust.": 37,
    "Pick up the pesticide atomizer in the garden and spray insectifuge to fully cover both potted plants in the garden.": 38,
    "In the garden, pick up the pesticide atomizer on the floor and spray pesticide onto both trees until each tree trunk is fully covered.": 39,
    "In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag.": 40,
    "From the kitchen refrigerator, take the cabbage and the chili, dice them on the chopping board with the knife, cook the diced cabbage and diced chili in the frying pan on the stove, and leave the cooked, diced cabbage and cooked, diced chili in the frying pan.": 41,
    "In the kitchen, take the Vidalia onion out of the sink, dice it on the chopping board with the paring knife, put the diced onion into the bowl on the countertop, then place both the paring knife and the chopping board into the sink.": 42,
    "From the refrigerator in the kitchen, take out the two bell peppers, the two beets, and the zucchini. Then, on either chopping board on the countertop, use the parer to dice all of them so that only diced bell pepper, diced beet, and diced zucchini remain. Make sure the refrigerator is closed when you finish.": 43,
    "Chop the four logs on the driveway in the garden into eight half logs using the axe and the chopping block.": 44,
    "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked.": 45,
    "Take the tray with six slices of bacon out of the refrigerator in the kitchen, cook all six slices in the frying pan on the stove until they're cooked, and make sure the refrigerator is closed when you're done.": 46,
    "In the kitchen, take the two apple pies from the plates on the countertop, put each pie into a separate tupperware container taken from the cabinet, place both tupperwares inside the refrigerator, close the refrigerator, and leave them until the pies are frozen.": 47,
    "In the kitchen, open the refrigerator and the cabinet. Take the steak and the pineapple out of the refrigerator and take two bowls from the cabinet. On the chopping board on the countertop, use the carving knife to dice the steak and to dice the pineapple. Put only the diced steak into one bowl and only the diced pineapple into the other bowl - do not mix them. Place both bowls back inside the cabinet, then close the refrigerator and close the cabinet.": 48,
    "Make a pizza on the cookie sheet in the kitchen: take the grated cheese, the four pieces of pepperoni, and the two mushrooms from their tupperware containers in the refrigerator; chop the Vidalia onion on the chopping board with the knife, and also chop the two whole mushrooms in half; top the pizza dough that's already on the cookie sheet with the cheese, pepperoni, halved mushrooms, and chopped onion; bake it in the oven until it becomes a pizza, and leave the finished pizza on the cookie sheet.": 49,
}
