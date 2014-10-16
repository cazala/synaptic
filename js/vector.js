function Vector(x, y)
{
	this.x = x;
	this.y = y;
}

Vector.prototype = {
	set: function(x, y)
	{
		this.x = x;
		this.y = y;

		return this;
	},
	add: function(v)
	{
		this.x += v.x;
		this.y += v.y;

		return this;
	},
	sub: function(v)
	{
		this.x -= v.x;
		this.y -= v.y;

		return this;
	},
	mul: function(s)
	{
		this.x *= s;
		this.y *= s;

		return this;
	},
	div: function(s)
	{
		!s && console.log("Division by zero!");

		this.x /= s;
		this.y /= s;

		return this;
	},
	mag: function(){
		return Math.sqrt(this.x * this.x + this.y * this.y);
	},
	normalize: function()
	{
		var mag = this.mag();
		mag && this.div(mag);
		return this;
	},
	angle: function()
	{
		return Math.atan2(this.y, this.x);
	},
	setMag: function(m)
	{
		var angle = this.angle();
		this.x = m * Math.cos(angle);
		this.y = m * Math.sin(angle);
		return this;
	},
	setAngle: function(a)
	{
		var mag = this.mag();
		this.x = mag * Math.cos(a);
		this.y = mag * Math.sin(a);
		return this;
	},
	rotate: function(a)
	{
		this.setAngle(this.angle() + a);
		return this;
	},
	limit: function(l)
	{
		var mag = this.mag();
		if(mag > l)
			this.setMag(l);
		return this;
	},
	angleBetween: function(v)
	{
		return this.angle() - v.angle();
	},
	dot: function(v)
	{
		return this.x * v.x + this.y * v.y;
	},
	lerp: function(v, amt)
	{
		this.x += (v.x - this.x) * amt;
		this.y += (v.y - this.y) * amt;
		return this;
	},
	dist: function(v)
	{
		var dx = this.x - v.x;
		var dy = this.y - v.y;
		return Math.sqrt(dx * dx + dy * dy);
	},
	copy: function()
	{
		return new Vector(this.x, this.y);
	},
	random: function(){
		this.set(1,1);
		this.setAngle(Math.random() * Math.PI * 2);
		return this;
	}
}