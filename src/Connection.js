export let connections = 0;

export default class Connection {
  constructor(from, to, weight) {
    if (!from || !to || !weight)
      throw new Error("Connection Error: Invalid neurons");

    this.ID = Connection.uid();
    this.from = from;
    this.to = to;
    this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 : weight;
    this.gain = 1;
    this.gater = null;
  }

  static uid() {
    return connections++;
  }
}
