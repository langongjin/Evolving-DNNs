import objectNEAT_2classRGB

save_path = './train2000-2/'
number_of_generations = 2001
testset_size = 0.25
objectNEAT_2classRGB.run_objectrecognition(number_of_generations, save_path, testset_size)