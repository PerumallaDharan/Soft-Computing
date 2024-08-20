import random, csv

inp = int(input("Enter the number of rows: "))

with open('Student_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Writing header row
    writer.writerow(['st_id', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'])

    for i in range(inp):
        # Generating a row with st_id and six random numbers
        row = [i + 1]
        row.extend(random.randint(0, 100) for _ in range(6))
        
        # Writing the row to the CSV file
        writer.writerow(row)
        
        # Printing the same row to the console
        print(" ".join(map(str, row[1:])))
