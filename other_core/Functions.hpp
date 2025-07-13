int col(int square){
    return square&7;
}

int row(int square){
    return square >> 3;
}

int color(int piece){
    return piece%2;
}

int type(int piece){
    return (piece-(piece%2));
}
