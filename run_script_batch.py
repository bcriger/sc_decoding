import os

if __name__ == '__main__':
    from sys import argv
    err_lo, err_hi = map(float, argv[1:3])
    n_points = int(argv[3])
    dists = list(map(int, argv[4].split()))
    n_trials = int(argv[5])
    flnm = argv[6]
    sim_type = argv[7] if len(argv) > 7 else 'iidxz'
    start_dx, end_dx = map(int,argv[8:10])

    str_dists = "'" + argv[4] + "'"

    for job_num in range(start_dx, end_dx + 1):
        new_argv = ["python", "run_script.py"] + argv[1:4] + [str_dists] + [argv[5]] + [flnm + '_' + str(job_num) + ".pkl"] + [sim_type] + [" &"]
        print ' '.join(new_argv)
        os.system(' '.join(new_argv))
