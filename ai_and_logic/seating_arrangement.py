SEATING_ARRANGEMENT = {
    1: "P001",
    2: "P002",
    3: "P003",
    4: "P004",
    5: "P005",
    6: "P006",
    7: "P007",
    8: "P008",
}


def get_student_id(bench_no):
    return SEATING_ARRANGEMENT.get(bench_no)


def get_bench_no(student_id):
    for bench_no, participant_id in SEATING_ARRANGEMENT.items():
        if participant_id == student_id:
            return bench_no

    return None