def main(num_slices, w, h):
    print("width:", w, ", height:", h, ", number of slices:", num_slices)
    ctr = 0
    for i in range(num_slices):
        for j in range(w):
            for k in range(h):
                idx1 = j * h + k
                idx2 = (k * w + j) * num_slices + i
                print("\t ", ctr, ") i:", i, ", j:", j, ", k:", k, " --> ", "idx1:", idx1, ", idx2:", idx2)
                ctr = ctr + 1
        print("--")    


if __name__ == "__main__":
    main(2,3,4)